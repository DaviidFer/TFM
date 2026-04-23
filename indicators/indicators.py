"""
Módulo de indicadores técnicos y generador de features predictoras.

La librería queda cerrada a 11 familias de indicadores, seleccionadas para el
TFM por su justificación académica:
Momentum, ROC, RSI, Stoch, WPR, CCI, BullsPower, BearsPower, DeMarker, RVI y DPO.
"""
import re

import numpy as np
import pandas as pd

NON_PREDICTIVE_COLUMNS = ("open", "high", "low", "close")
ACTIVE_INDICATOR_FAMILIES = (
    "Momentum",
    "ROC",
    "RSI",
    "Stoch",
    "WPR",
    "CCI",
    "BullsPower",
    "BearsPower",
    "DeMarker",
    "RVI",
    "DPO",
)
ACTIVE_INDICATOR_PREFIXES = tuple(f"{name}_" for name in ACTIVE_INDICATOR_FAMILIES)
_ALLOWED_FEATURE_RE = re.compile(
    r"^(Momentum|ROC|RSI|Stoch|WPR|CCI|BullsPower|BearsPower|DeMarker|RVI|DPO)_(\d+)$"
)


# =========================================================
# FUNCIONES BASE / HELPERS
# =========================================================
def safe_div(a, b):
    if isinstance(b, pd.Series):
        b = b.replace(0, np.nan)
    else:
        b = np.where(b == 0, np.nan, b)
    return a / b


# =========================================================
# INDICADORES ACTIVOS
# =========================================================
def ema_mt5(close: pd.Series, n: int) -> pd.Series:
    if n <= 0:
        raise ValueError("n debe ser > 0")

    arr = pd.Series(close, dtype=float).to_numpy()
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)

    if arr.shape[0] >= n:
        alpha = 2.0 / (n + 1.0)
        out[n - 1] = np.mean(arr[:n])
        for i in range(n, arr.shape[0]):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]

    return pd.Series(out, index=close.index, dtype="float32")


def bears_power_mt5(low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    return (pd.Series(low, dtype=float) - ema_mt5(close, n)).astype("float32")


def bulls_power_mt5(high: pd.Series, close: pd.Series, n: int) -> pd.Series:
    return (pd.Series(high, dtype=float) - ema_mt5(close, n)).astype("float32")


def wpr_mt5(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    high = pd.Series(high, dtype=float)
    low = pd.Series(low, dtype=float)
    close = pd.Series(close, dtype=float)

    hh = high.rolling(window=n, min_periods=n).max()
    ll = low.rolling(window=n, min_periods=n).min()
    denom = hh - ll

    out = -100.0 * (hh - close) / denom
    out[denom == 0] = np.nan
    return out.astype("float32")


def demarker_mt5(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    high = pd.Series(high, dtype=float)
    low = pd.Series(low, dtype=float)

    dh = high.diff()
    dl = low.diff()

    demax = np.where(dh > 0, dh, 0.0)
    demin = np.where(-dl > 0, -dl, 0.0)

    demax_sma = pd.Series(demax, index=high.index).rolling(n, min_periods=n).mean()
    demin_sma = pd.Series(demin, index=low.index).rolling(n, min_periods=n).mean()
    denom = demax_sma + demin_sma

    out = demax_sma / denom
    out[denom == 0] = np.nan
    return out.astype("float32")


def momentum_mt5(close: pd.Series, n: int) -> pd.Series:
    close = pd.Series(close, dtype=float)
    return (close - close.shift(n)).astype("float32")


def cci_mt5(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    high = pd.Series(high, dtype=float)
    low = pd.Series(low, dtype=float)
    close = pd.Series(close, dtype=float)

    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(n, min_periods=n).mean()
    mad = tp.rolling(n, min_periods=n).apply(
        lambda x: np.mean(np.abs(x - x.mean())),
        raw=True
    )

    denom = 0.015 * mad
    out = (tp - sma_tp) / denom
    out[denom == 0] = np.nan
    return out.astype("float32")


def stochastic_mt5_k_only(high: pd.Series, low: pd.Series, close: pd.Series,
                          k_period: int, slowing: int = 2) -> pd.Series:
    high = pd.Series(high, dtype=float)
    low = pd.Series(low, dtype=float)
    close = pd.Series(close, dtype=float)

    hh = high.rolling(k_period, min_periods=k_period).max()
    ll = low.rolling(k_period, min_periods=k_period).min()

    num = close - ll
    den = hh - ll

    sum_num = num.rolling(slowing, min_periods=slowing).sum()
    sum_den = den.rolling(slowing, min_periods=slowing).sum()

    k_main = 100.0 * (sum_num / sum_den)
    k_main[sum_den == 0] = 100.0
    return k_main.astype("float32")


def rsi_mt5(price: pd.Series, n: int) -> pd.Series:
    p = pd.Series(price, dtype=float).to_numpy()
    L = len(p)

    rsi = np.zeros(L, dtype=np.float64)
    pos = np.zeros(L, dtype=np.float64)
    neg = np.zeros(L, dtype=np.float64)

    if L == 0 or n < 1:
        return pd.Series(rsi, index=price.index, dtype="float32")

    if L <= n:
        return pd.Series(rsi, index=price.index, dtype="float32")

    sum_pos = 0.0
    sum_neg = 0.0

    for i in range(1, n + 1):
        diff = p[i] - p[i - 1]
        if diff > 0:
            sum_pos += diff
        elif diff < 0:
            sum_neg += -diff

        pos[i] = 0.0
        neg[i] = 0.0
        rsi[i] = 0.0

    pos[n] = sum_pos / n
    neg[n] = sum_neg / n

    if neg[n] != 0.0:
        rsi[n] = 100.0 - (100.0 / (1.0 + pos[n] / neg[n]))
    else:
        rsi[n] = 100.0 if pos[n] != 0.0 else 50.0

    for i in range(n + 1, L):
        diff = p[i] - p[i - 1]
        gain = diff if diff > 0.0 else 0.0
        loss = -diff if diff < 0.0 else 0.0

        pos[i] = (pos[i - 1] * (n - 1) + gain) / n
        neg[i] = (neg[i - 1] * (n - 1) + loss) / n

        if neg[i] != 0.0:
            rsi[i] = 100.0 - 100.0 / (1.0 + pos[i] / neg[i])
        else:
            rsi[i] = 100.0 if pos[i] != 0.0 else 50.0

    return pd.Series(rsi, index=price.index, dtype="float32")


def roc_mt5(price: pd.Series, n: int) -> pd.Series:
    p = pd.Series(price, dtype=float)
    prev = p.shift(n)
    out = 100.0 * safe_div((p - prev), prev)
    return out.astype("float32")


def dpo_mt5(price: pd.Series, detrend_period: int) -> pd.Series:
    p = pd.Series(price, dtype=float)
    sma_n = p.rolling(detrend_period, min_periods=detrend_period).mean()
    shift_n = detrend_period // 2 + 1
    dpo = p - sma_n.shift(shift_n)
    return dpo.astype("float32")


def rvi_mt5_main(df_ohlc: pd.DataFrame, n: int) -> pd.Series:
    open_ = pd.Series(df_ohlc["open"], dtype=float)
    high = pd.Series(df_ohlc["high"], dtype=float)
    low = pd.Series(df_ohlc["low"], dtype=float)
    close = pd.Series(df_ohlc["close"], dtype=float)

    num = (
        (close - open_)
        + 2.0 * (close.shift(1) - open_.shift(1))
        + 2.0 * (close.shift(2) - open_.shift(2))
        + (close.shift(3) - open_.shift(3))
    ) / 6.0
    den = (
        (high - low)
        + 2.0 * (high.shift(1) - low.shift(1))
        + 2.0 * (high.shift(2) - low.shift(2))
        + (high.shift(3) - low.shift(3))
    ) / 6.0

    num_avg = num.rolling(n, min_periods=n).mean()
    den_avg = den.rolling(n, min_periods=n).mean()
    rvi = safe_div(num_avg, den_avg)
    return pd.Series(rvi, index=df_ohlc.index, dtype="float32")


def _normalize_periods(periods) -> list[int]:
    normalized = sorted({int(p) for p in periods if int(p) > 0})
    if not normalized:
        raise ValueError("Debe indicarse al menos un periodo positivo.")
    return normalized


def validate_feature_frame(
    df_features: pd.DataFrame,
    *,
    periods,
    include_rvi: bool = True,
    raise_on_error: bool = True,
) -> dict[str, object]:
    duplicates = df_features.columns[df_features.columns.duplicated()].tolist()
    predictive_cols = [c for c in df_features.columns if c not in NON_PREDICTIVE_COLUMNS]
    invalid_predictive = [c for c in predictive_cols if _ALLOWED_FEATURE_RE.match(str(c)) is None]

    periods = _normalize_periods(periods)
    expected_cols = {
        f"{family}_{period}"
        for family in ACTIVE_INDICATOR_FAMILIES
        if include_rvi or family != "RVI"
        for period in periods
    }
    missing_expected = sorted(expected_cols.difference(predictive_cols))
    unexpected_predictive = sorted(set(predictive_cols).difference(expected_cols))

    warmup = max(periods) + max(4, 1)
    tail_slice = df_features.iloc[min(len(df_features), warmup):]
    all_nan_after_warmup = []
    if not tail_slice.empty:
        for col in predictive_cols:
            if tail_slice[col].notna().sum() == 0:
                all_nan_after_warmup.append(col)

    result = {
        "predictive_columns": predictive_cols,
        "duplicates": duplicates,
        "invalid_predictive": invalid_predictive,
        "missing_expected": missing_expected,
        "unexpected_predictive": unexpected_predictive,
        "all_nan_after_warmup": all_nan_after_warmup,
    }
    errors = {
        key: value for key, value in result.items()
        if key != "predictive_columns" and value
    }
    if raise_on_error and errors:
        raise ValueError(f"Feature library validation failed: {errors}")
    return result


# =========================================================
# GENERADOR DE FEATURES
# =========================================================
def build_feature_library(
    data_ohlc: pd.DataFrame,
    periods=range(2, 101, 2),
    breakout_periods=(2, 3, 4, 5, 7, 10, 14, 20, 28, 50, 100),
    seq_periods=(2, 3, 4, 5),
    slowing=2,
    include_rvi=True,
    shift_features=1,
    dropna=True
):
    df = data_ohlc.copy()

    for c in NON_PREDICTIVE_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    feat = {}
    periods = _normalize_periods(periods)

    def add_feature(name: str, s, dtype="float32"):
        s = pd.Series(s, index=df.index)
        if dtype == "int8":
            feat[name] = s.fillna(False).astype("int8")
        else:
            feat[name] = pd.to_numeric(s, errors="coerce").astype("float32")

    # `breakout_periods` y `seq_periods` se conservan en la firma solo por compatibilidad
    # con llamadas existentes; la librería predictora queda cerrada a las 11 familias.
    for n in periods:
        add_feature(f"BearsPower_{n}", bears_power_mt5(low, close, n))
        add_feature(f"BullsPower_{n}", bulls_power_mt5(high, close, n))
        add_feature(f"DeMarker_{n}", demarker_mt5(high, low, n))
        add_feature(f"Momentum_{n}", momentum_mt5(close, n))
        add_feature(f"WPR_{n}", wpr_mt5(high, low, close, n))
        add_feature(f"CCI_{n}", cci_mt5(high, low, close, n))
        add_feature(f"RSI_{n}", rsi_mt5(close, n))
        add_feature(f"Stoch_{n}", stochastic_mt5_k_only(high, low, close, k_period=n, slowing=slowing))
        add_feature(f"ROC_{n}", roc_mt5(close, n))
        add_feature(f"DPO_{n}", dpo_mt5(close, n))

        if include_rvi:
            add_feature(f"RVI_{n}", rvi_mt5_main(df[["open", "high", "low", "close"]], n))

    # -----------------------------------------------------
    # UNIÓN FINAL
    # -----------------------------------------------------
    feat_df = pd.DataFrame(feat, index=df.index)

    # Leak-safe alignment for open-to-open target:
    # if Target[t] = (Open[t+1]-Open[t])/Open[t], features at row t must come
    # from information available no later than t-1 close.
    if int(shift_features) != 0:
        feat_df = feat_df.shift(int(shift_features))

    df_out = pd.concat([df, feat_df], axis=1)
    df_out = df_out.replace([np.inf, -np.inf], np.nan)
    validate_feature_frame(
        df_out,
        periods=periods,
        include_rvi=include_rvi,
    )

    if dropna:
        df_out = df_out.dropna().copy()

    return df_out
