from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yfinance as yf
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class YahooCompanyData:
    ticker: str
    info: Dict[str, Any]
    financials: Optional[pd.DataFrame]
    balance_sheet: Optional[pd.DataFrame]
    cashflow: Optional[pd.DataFrame]


class YahooProvider:
    """Thin wrapper around yfinance with caching and basic retries."""

    def __init__(self, ttl_seconds: int = 6 * 3600, max_cache_items: int = 256):
        self._cache = TTLCache(maxsize=max_cache_items, ttl=ttl_seconds)

    def _cache_key(self, ticker: str) -> str:
        return ticker.upper().strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.8, min=0.8, max=6))
    def fetch_company_data(self, ticker: str) -> YahooCompanyData:
        key = self._cache_key(ticker)
        if key in self._cache:
            return self._cache[key]

        t = yf.Ticker(key)

        # yfinance fields can raise intermittently; keep it defensive.
        info: Dict[str, Any] = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}

        def safe_df(getter) -> Optional[pd.DataFrame]:
            try:
                df = getter()
                if df is None or (hasattr(df, "empty") and df.empty):
                    return None
                return df
            except Exception:
                return None

        # Annual statements (columns are periods)
        financials = safe_df(lambda: t.financials)
        balance_sheet = safe_df(lambda: t.balance_sheet)
        cashflow = safe_df(lambda: t.cashflow)

        data = YahooCompanyData(
            ticker=key,
            info=info,
            financials=financials,
            balance_sheet=balance_sheet,
            cashflow=cashflow,
        )
        self._cache[key] = data
        return data


def _pick_row(df: Optional[pd.DataFrame], candidates: Tuple[str, ...]) -> Optional[pd.Series]:
    if df is None:
        return None
    for name in candidates:
        if name in df.index:
            s = df.loc[name]
            try:
                return pd.to_numeric(s, errors="coerce")
            except Exception:
                return s
    # Try loose matching
    lowered = {str(i).lower(): i for i in df.index}
    for name in candidates:
        k = name.lower()
        if k in lowered:
            s = df.loc[lowered[k]]
            try:
                return pd.to_numeric(s, errors="coerce")
            except Exception:
                return s
    return None


def extract_statement_series(data: YahooCompanyData) -> Dict[str, Optional[pd.Series]]:
    """Extract commonly-used line items as numeric Series indexed by fiscal period columns."""
    fin = data.financials
    cf = data.cashflow
    bs = data.balance_sheet

    revenue = _pick_row(fin, ("Total Revenue", "TotalRevenue"))
    ebit = _pick_row(fin, ("Ebit", "EBIT", "Operating Income", "OperatingIncome"))
    pretax = _pick_row(fin, ("Pretax Income", "PretaxIncome", "Income Before Tax", "IncomeBeforeTax"))
    tax = _pick_row(fin, ("Tax Provision", "TaxProvision", "Income Tax Expense", "IncomeTaxExpense"))

    da = _pick_row(cf, ("Depreciation", "Depreciation And Amortization", "DepreciationAndAmortization"))
    capex = _pick_row(cf, ("Capital Expenditures", "CapitalExpenditures"))
    wc_item = _pick_row(cf, ("Change In Working Capital", "ChangeInWorkingCapital"))

    cash = _pick_row(bs, ("Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash", "Cash And Short Term Investments"))

    debt = None
    if bs is not None:
        debt_parts = []
        for nm in (
            "Short Long Term Debt",
            "ShortLongTermDebt",
            "Long Term Debt",
            "LongTermDebt",
            "Long Term Debt And Capital Lease Obligation",
            "LongTermDebtAndCapitalLeaseObligation",
            "Current Debt",
            "CurrentDebt",
        ):
            s = _pick_row(bs, (nm,))
            if s is not None:
                debt_parts.append(s)
        if debt_parts:
            try:
                debt = sum(debt_parts)
            except Exception:
                debt = debt_parts[0]

    return {
        "revenue": revenue,
        "ebit": ebit,
        "pretax": pretax,
        "tax": tax,
        "da": da,
        "capex": capex,
        "wc_item": wc_item,
        "cash": cash,
        "debt": debt,
    }
