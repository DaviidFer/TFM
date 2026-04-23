from pathlib import Path
import sys
import subprocess
import time
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")

# =========================
# CONFIGURACIÓN
# =========================
DATA_ROOT = Path("datos")
STOCKS_DIR = DATA_ROOT / "Stocks"
ETFS_DIR = DATA_ROOT / "ETFs"
UNIVERSE_DIR = DATA_ROOT / "_universe"

# Fecha inicial para activos nuevos
DEFAULT_START_DATE = "2000-01-01"

# True  -> reconstruye el universo exacto desde MT5
# False -> usa los ficheros ya congelados en datos/_universe/
REBUILD_UNIVERSE_FROM_MT5 = True

# Opcional: si MT5 no se detecta automáticamente, pon aquí la ruta al terminal64.exe
# Ejemplo:
# MT5_PATH = r"C:\Program Files\Darwinex MetaTrader 5\terminal64.exe"
MT5_PATH = None

# Pausas / reintentos para Yahoo
SLEEP_BETWEEN_SYMBOLS = 0.25
MAX_RETRIES_PER_SYMBOL = 4
BACKOFF_BASE_SECONDS = 1.5

# Si quieres usar listas manuales en vez de MT5, pégalas aquí
MANUAL_DZ_STOCKS = []
MANUAL_DZ_ETFS = []


def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


def _import_dependencies():
    ensure_package("pandas")
    ensure_package("yfinance")
    try:
        import MetaTrader5 as mt5  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "MetaTrader5"])

    import pandas as pd  # type: ignore
    import yfinance as yf  # type: ignore
    import MetaTrader5 as mt5  # type: ignore

    return pd, yf, mt5


def _prepare_directories():
    for p in [DATA_ROOT, STOCKS_DIR, ETFS_DIR, UNIVERSE_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def unique_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def read_txt_list(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_txt_list(path, items):
    items = unique_keep_order(sorted(items))
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(f"{x}\n")


def safe_filename(symbol):
    return (
        symbol.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )


def yahoo_symbol_candidates(raw_symbol):
    raw = str(raw_symbol).strip().upper()
    cands = [raw]

    if "." in raw:
        cands.append(raw.replace(".", "-"))

    if "/" in raw:
        cands.append(raw.replace("/", "-"))

    if "_" in raw:
        cands.append(raw.replace("_", "-"))

    for suf in [".US", ".USD", "_US", "_USD", ".CASH"]:
        if raw.endswith(suf):
            cands.append(raw[: -len(suf)])

    extra = []
    for x in cands:
        if "." in x:
            extra.append(x.replace(".", "-"))
        if "/" in x:
            extra.append(x.replace("/", "-"))
        if "_" in x:
            extra.append(x.replace("_", "-"))

    cands.extend(extra)
    return unique_keep_order(cands)


def connect_mt5(mt5, mt5_path=None):
    if mt5_path:
        ok = mt5.initialize(path=mt5_path)
    else:
        ok = mt5.initialize()

    if not ok:
        raise RuntimeError(f"No se pudo inicializar MT5. last_error={mt5.last_error()}")


def disconnect_mt5(mt5):
    try:
        mt5.shutdown()
    except Exception:
        pass


def classify_mt5_symbol(info_dict):
    name = str(info_dict.get("name", "")).strip()
    path = str(info_dict.get("path", "")).strip()
    description = str(info_dict.get("description", "")).strip()

    hay = f"{path} | {description}".lower()

    exclude_words = [
        "forex",
        "fx",
        "index",
        "indices",
        "commodity",
        "metal",
        "future",
        "futures",
        "crypto",
        "bond",
        "option",
    ]
    if any(w in hay for w in exclude_words):
        return None

    if "etf" in hay:
        return "ETF"

    stock_words = ["stock", "stocks", "share", "shares", "acciones", "equities", "equity"]
    if any(w in hay for w in stock_words):
        return "Stock"

    fallback_words = ["nasdaq", "nyse", "amex", "usa", "us "]
    if any(w in hay for w in fallback_words):
        return "Stock"

    return None


def build_universe_from_mt5(pd, mt5, stocks_txt, etfs_txt, universe_csv, mt5_raw_csv, mt5_path=None):
    connect_mt5(mt5, mt5_path)

    symbols = mt5.symbols_get()
    if symbols is None:
        disconnect_mt5(mt5)
        raise RuntimeError(f"mt5.symbols_get() devolvió None. last_error={mt5.last_error()}")

    raw_rows = []
    rows = []

    for s in symbols:
        d = s._asdict() if hasattr(s, "_asdict") else {}
        name = str(d.get("name", "")).strip()
        path = str(d.get("path", "")).strip()
        description = str(d.get("description", "")).strip()

        raw_rows.append({"name": name, "path": path, "description": description})

        asset_type = classify_mt5_symbol(d)
        if asset_type not in {"Stock", "ETF"}:
            continue

        if not name:
            continue

        rows.append(
            {"asset_type": asset_type, "symbol": name.upper(), "description": description, "path": path}
        )

    disconnect_mt5(mt5)

    raw_df = pd.DataFrame(raw_rows).drop_duplicates()
    raw_df.to_csv(mt5_raw_csv, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows).drop_duplicates(subset=["asset_type", "symbol"]).copy()
    if df.empty:
        raise RuntimeError(
            "No he podido clasificar símbolos de Stock/ETF desde MT5. "
            "Revisa datos/_universe/mt5_all_symbols_raw.csv para ver los paths reales "
            "y ajustar la heurística."
        )

    df = df.sort_values(["asset_type", "symbol"]).reset_index(drop=True)

    stocks = df.loc[df["asset_type"] == "Stock", "symbol"].tolist()
    etfs = df.loc[df["asset_type"] == "ETF", "symbol"].tolist()

    write_txt_list(stocks_txt, stocks)
    write_txt_list(etfs_txt, etfs)
    df.to_csv(universe_csv, index=False, encoding="utf-8-sig")

    return stocks, etfs, df


def load_or_build_universe(pd, stocks_txt, etfs_txt, universe_csv, mt5_raw_csv):
    if MANUAL_DZ_STOCKS or MANUAL_DZ_ETFS:
        stocks = unique_keep_order([x.upper() for x in MANUAL_DZ_STOCKS])
        etfs = unique_keep_order([x.upper() for x in MANUAL_DZ_ETFS])

        write_txt_list(stocks_txt, stocks)
        write_txt_list(etfs_txt, etfs)

        rows = [
            {"asset_type": "Stock", "symbol": s, "description": "", "path": "MANUAL"}
            for s in stocks
        ]
        rows += [
            {"asset_type": "ETF", "symbol": s, "description": "", "path": "MANUAL"}
            for s in etfs
        ]
        df = pd.DataFrame(rows)
        df.to_csv(universe_csv, index=False, encoding="utf-8-sig")
        return stocks, etfs, df

    if REBUILD_UNIVERSE_FROM_MT5:
        _, _, mt5 = _import_dependencies()
        stocks, etfs, df = build_universe_from_mt5(
            pd=pd,
            mt5=mt5,
            stocks_txt=stocks_txt,
            etfs_txt=etfs_txt,
            universe_csv=universe_csv,
            mt5_raw_csv=mt5_raw_csv,
            mt5_path=MT5_PATH,
        )
        return stocks, etfs, df

    stocks = read_txt_list(stocks_txt)
    etfs = read_txt_list(etfs_txt)

    if not stocks and not etfs:
        raise RuntimeError(
            "No existen listas congeladas y REBUILD_UNIVERSE_FROM_MT5=False. "
            "Pon REBUILD_UNIVERSE_FROM_MT5=True o rellena MANUAL_DZ_STOCKS / MANUAL_DZ_ETFS."
        )

    rows = [
        {"asset_type": "Stock", "symbol": s, "description": "", "path": "LOCAL_TXT"}
        for s in stocks
    ]
    rows += [
        {"asset_type": "ETF", "symbol": s, "description": "", "path": "LOCAL_TXT"}
        for s in etfs
    ]
    df = pd.DataFrame(rows)
    df.to_csv(universe_csv, index=False, encoding="utf-8-sig")
    return stocks, etfs, df


def fetch_yahoo_history_1d(pd, yf, raw_symbol, start_date, end_date):
    last_error = None

    for y_symbol in yahoo_symbol_candidates(raw_symbol):
        for attempt in range(1, MAX_RETRIES_PER_SYMBOL + 1):
            try:
                df = yf.download(
                    tickers=y_symbol,
                    start=start_date,
                    end=end_date,
                    interval="1d",
                    auto_adjust=False,
                    actions=True,
                    progress=False,
                    threads=False,
                )

                if df is None:
                    df = pd.DataFrame()

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if not df.empty:
                    df = df.reset_index()
                    if "Date" not in df.columns:
                        df = df.rename(columns={df.columns[0]: "Date"})

                    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                    df["RawSymbol"] = raw_symbol
                    df["YahooSymbol"] = y_symbol

                    for col in [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Adj Close",
                        "Volume",
                        "Dividends",
                        "Stock Splits",
                    ]:
                        if col not in df.columns:
                            df[col] = pd.NA

                    df = df[
                        [
                            "Date",
                            "Open",
                            "High",
                            "Low",
                            "Close",
                            "Adj Close",
                            "Volume",
                            "Dividends",
                            "Stock Splits",
                            "RawSymbol",
                            "YahooSymbol",
                        ]
                    ].copy()

                    return df, y_symbol, None

                last_error = f"Descarga vacía para {y_symbol}"

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"

            sleep_s = BACKOFF_BASE_SECONDS * attempt
            time.sleep(sleep_s)

    return pd.DataFrame(), None, last_error


def update_one_symbol(pd, yf, raw_symbol, asset_type, folder, default_start_date=DEFAULT_START_DATE):
    csv_path = folder / f"{safe_filename(raw_symbol)}.csv"

    old_df = pd.DataFrame()
    if csv_path.exists():
        try:
            old_df = pd.read_csv(csv_path, parse_dates=["Date"])
            old_df["Date"] = pd.to_datetime(old_df["Date"]).dt.tz_localize(None)
        except Exception:
            old_df = pd.DataFrame()

    if old_df.empty:
        start_date = pd.Timestamp(default_start_date)
    else:
        last_dt = pd.to_datetime(old_df["Date"]).max()
        start_date = last_dt + pd.Timedelta(days=1)

    end_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

    if start_date >= end_date:
        return {
            "asset_type": asset_type,
            "symbol": raw_symbol,
            "status": "up_to_date",
            "rows_added": 0,
            "yahoo_symbol": (
                old_df["YahooSymbol"].dropna().iloc[-1]
                if ("YahooSymbol" in old_df.columns and len(old_df) > 0)
                else None
            ),
            "error": None,
        }

    new_df, yahoo_symbol, err = fetch_yahoo_history_1d(
        pd=pd,
        yf=yf,
        raw_symbol=raw_symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    if new_df.empty:
        if not old_df.empty:
            return {
                "asset_type": asset_type,
                "symbol": raw_symbol,
                "status": "up_to_date",
                "rows_added": 0,
                "yahoo_symbol": (
                    old_df["YahooSymbol"].dropna().iloc[-1]
                    if ("YahooSymbol" in old_df.columns and old_df["YahooSymbol"].notna().any())
                    else yahoo_symbol
                ),
                "error": None,
            }
        else:
            return {
                "asset_type": asset_type,
                "symbol": raw_symbol,
                "status": "failed",
                "rows_added": 0,
                "yahoo_symbol": yahoo_symbol,
                "error": err,
            }

    combined = pd.concat([old_df, new_df], ignore_index=True)

    for col in [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "RawSymbol",
        "YahooSymbol",
    ]:
        if col not in combined.columns:
            combined[col] = pd.NA

    combined["Date"] = pd.to_datetime(combined["Date"]).dt.tz_localize(None)
    combined["RawSymbol"] = combined["RawSymbol"].fillna(raw_symbol)
    if yahoo_symbol:
        combined["YahooSymbol"] = combined["YahooSymbol"].fillna(yahoo_symbol)

    combined = combined[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "Dividends",
            "Stock Splits",
            "RawSymbol",
            "YahooSymbol",
        ]
    ].copy()

    before = 0 if old_df.empty else len(old_df)
    combined = (
        combined.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    after = len(combined)
    rows_added = after - before

    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {
        "asset_type": asset_type,
        "symbol": raw_symbol,
        "status": "updated" if rows_added > 0 else "up_to_date",
        "rows_added": int(max(rows_added, 0)),
        "yahoo_symbol": yahoo_symbol,
        "error": None,
    }


def update_group(pd, yf, symbols, asset_type, folder):
    results = []
    total = len(symbols)

    for symbol in symbols:
        res = update_one_symbol(pd, yf, symbol, asset_type, folder)
        results.append(res)
        time.sleep(SLEEP_BETWEEN_SYMBOLS)

    return pd.DataFrame(results)


def run_data_download():
    pd, yf, _ = _import_dependencies()
    _prepare_directories()

    stocks_txt = UNIVERSE_DIR / "dz_stocks.txt"
    etfs_txt = UNIVERSE_DIR / "dz_etfs.txt"
    universe_csv = UNIVERSE_DIR / "dz_universe_full.csv"
    mt5_raw_csv = UNIVERSE_DIR / "mt5_all_symbols_raw.csv"

    stocks_list, etfs_list, universe_df = load_or_build_universe(
        pd=pd,
        stocks_txt=stocks_txt,
        etfs_txt=etfs_txt,
        universe_csv=universe_csv,
        mt5_raw_csv=mt5_raw_csv,
    )

    stocks_results = update_group(pd, yf, stocks_list, "Stock", STOCKS_DIR)
    etfs_results = update_group(pd, yf, etfs_list, "ETF", ETFS_DIR)

    all_results = pd.concat([stocks_results, etfs_results], ignore_index=True)
    all_results.to_csv(DATA_ROOT / "update_log.csv", index=False, encoding="utf-8-sig")

    failed = all_results[all_results["status"] == "failed"].copy()
    if not failed.empty:
        failed.to_csv(DATA_ROOT / "failed_symbols.csv", index=False, encoding="utf-8-sig")

    return universe_df, all_results, failed

