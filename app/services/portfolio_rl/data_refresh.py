from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pandas as pd

from app.execution.local_data_provider import LocalMarketDataProvider
from data_download.download import run_data_download


@dataclass(frozen=True)
class OHLCRefreshResult:
    cutoff_date: str
    refreshed_symbols: List[str]
    n_requested_symbols: int
    n_refreshed_symbols: int
    status: str
    metadata: Dict[str, Any]


class PortfolioOHLCRefreshService:
    def __init__(self, local_data_provider: LocalMarketDataProvider) -> None:
        self.local_data_provider = local_data_provider

    def _resolve_cutoff_date(self, symbols: Iterable[str]) -> str:
        latest_dates: List[pd.Timestamp] = []
        for symbol in {str(s).upper() for s in symbols if str(s).strip()}:
            path = self.local_data_provider.get_csv_path(symbol)
            if not path:
                continue
            df = pd.read_csv(path)
            date_col = "Date" if "Date" in df.columns else "date"
            if date_col not in df.columns:
                continue
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not dates.empty:
                latest_dates.append(pd.Timestamp(dates.max()).normalize())
        if not latest_dates:
            return pd.Timestamp.utcnow().normalize().date().isoformat()
        return max(latest_dates).date().isoformat()

    def refresh(self, symbols: Iterable[str]) -> OHLCRefreshResult:
        requested = {str(s).upper() for s in symbols if str(s).strip()}
        universe_df, all_results, failed = run_data_download(symbols=requested)
        self.local_data_provider.refresh_symbol_registry()
        refreshed_symbols = sorted(
            {
                str(row["symbol"]).upper()
                for _, row in all_results.iterrows()
                if str(row.get("symbol") or "").upper() in requested and str(row.get("status") or "") != "failed"
            }
        )
        cutoff_date = self._resolve_cutoff_date(requested or universe_df.get("symbol", []).tolist())
        return OHLCRefreshResult(
            cutoff_date=cutoff_date,
            refreshed_symbols=refreshed_symbols,
            n_requested_symbols=len(requested),
            n_refreshed_symbols=len(refreshed_symbols),
            status="error" if not failed.empty and not refreshed_symbols else "ok",
            metadata={
                "failed_symbols": failed["symbol"].astype(str).tolist() if not failed.empty and "symbol" in failed.columns else [],
                "update_rows": all_results.to_dict(orient="records"),
            },
        )
