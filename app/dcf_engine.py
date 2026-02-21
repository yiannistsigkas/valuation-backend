from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class HistoricalMetrics:
    periods: List[str]
    revenue: List[Optional[float]]
    ebit: List[Optional[float]]
    tax_rate: Optional[float]
    da: List[Optional[float]]
    capex: List[Optional[float]]
    wc_item: List[Optional[float]]


@dataclass
class ForecastDrivers:
    years: int
    revenue_growth: List[float]
    ebit_margin: List[float]
    da_pct_rev: float
    capex_pct_rev: float
    wc_item_pct_rev: float


@dataclass
class DCFResult:
    value_per_share: Optional[float]
    enterprise_value: Optional[float]
    equity_value: Optional[float]
    pv_fcfs: Optional[float]
    pv_terminal: Optional[float]


def _period_labels_from_series(s: pd.Series) -> List[str]:
    cols = list(s.index)
    # yfinance often uses Timestamps
    def to_label(x):
        try:
            if hasattr(x, "date"):
                return str(x.date())
            return str(x)
        except Exception:
            return str(x)

    # Most recent first (timestamps compare)
    try:
        cols_sorted = sorted(cols, reverse=True)
    except Exception:
        cols_sorted = cols
    return [to_label(c) for c in cols_sorted]


def _values_from_series(s: Optional[pd.Series], periods_sorted: List) -> List[Optional[float]]:
    if s is None:
        return [None for _ in periods_sorted]
    vals = []
    for p in periods_sorted:
        v = s.get(p, np.nan)
        try:
            fv = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
        except Exception:
            fv = None
        vals.append(fv)
    return vals


def build_historical(series_map: Dict[str, Optional[pd.Series]], max_years: int = 4) -> HistoricalMetrics:
    # Determine periods from whichever exists
    candidate = None
    for k in ("revenue", "ebit", "da", "capex"):
        if series_map.get(k) is not None:
            candidate = series_map[k]
            break
    if candidate is None:
        return HistoricalMetrics(periods=[], revenue=[], ebit=[], tax_rate=None, da=[], capex=[], wc_item=[])

    periods = list(candidate.index)
    try:
        periods_sorted = sorted(periods, reverse=True)
    except Exception:
        periods_sorted = periods
    periods_sorted = periods_sorted[:max_years]

    labels = []
    for p in periods_sorted:
        try:
            labels.append(str(p.date()) if hasattr(p, "date") else str(p))
        except Exception:
            labels.append(str(p))

    rev = _values_from_series(series_map.get("revenue"), periods_sorted)
    ebit = _values_from_series(series_map.get("ebit"), periods_sorted)
    da = _values_from_series(series_map.get("da"), periods_sorted)
    capex_raw = _values_from_series(series_map.get("capex"), periods_sorted)
    # Capex often negative in CF. Store as positive magnitude.
    capex = [abs(v) if v is not None else None for v in capex_raw]
    wc_item = _values_from_series(series_map.get("wc_item"), periods_sorted)

    # Tax rate from most recent values
    tax_rate = None
    pretax_s = series_map.get("pretax")
    tax_s = series_map.get("tax")
    if pretax_s is not None and tax_s is not None:
        # use most recent period that has both
        for p in periods_sorted:
            pretax = pretax_s.get(p, np.nan)
            tax = tax_s.get(p, np.nan)
            try:
                pretax_f = float(pretax)
                tax_f = float(tax)
                if pretax_f and pretax_f > 0:
                    tr = tax_f / pretax_f
                    if np.isfinite(tr):
                        tax_rate = float(np.clip(tr, 0.0, 0.4))
                        break
            except Exception:
                continue

    return HistoricalMetrics(periods=labels, revenue=rev, ebit=ebit, tax_rate=tax_rate, da=da, capex=capex, wc_item=wc_item)


def _safe_last(xs: List[Optional[float]]) -> Optional[float]:
    for v in xs:
        if v is not None and np.isfinite(v):
            return float(v)
    return None


def _avg_ratio(num: List[Optional[float]], den: List[Optional[float]], n: int = 3, default: float = 0.0) -> float:
    pairs = []
    for a, b in zip(num[:n], den[:n]):
        if a is None or b is None:
            continue
        if b == 0:
            continue
        pairs.append(a / b)
    if not pairs:
        return default
    r = float(np.mean(pairs))
    # keep in a sane range
    return float(np.clip(r, -0.5, 0.5))


def infer_drivers(hist: HistoricalMetrics, years: int, terminal_growth: float) -> ForecastDrivers:
    base_rev = _safe_last(hist.revenue)
    # historical CAGR from last 3 points if available
    g_hist = None
    try:
        revs = [v for v in hist.revenue if v is not None]
        if len(revs) >= 3 and revs[0] and revs[2] and revs[2] > 0:
            g_hist = (revs[0] / revs[2]) ** (1 / 2) - 1
    except Exception:
        g_hist = None

    base_g = float(g_hist) if g_hist is not None and np.isfinite(g_hist) else 0.06
    base_g = float(np.clip(base_g, -0.05, 0.20))

    # fade growth toward terminal growth over forecast period
    revenue_growth = list(np.linspace(base_g, terminal_growth, years).astype(float))

    # EBIT margin from most recent
    last_rev = _safe_last(hist.revenue)
    last_ebit = _safe_last(hist.ebit)
    margin = 0.15
    if last_rev and last_ebit is not None and last_rev != 0:
        margin = float(np.clip(last_ebit / last_rev, -0.2, 0.5))
    ebit_margin = [margin for _ in range(years)]

    da_pct = abs(_avg_ratio(hist.da, hist.revenue, default=0.04))
    capex_pct = abs(_avg_ratio(hist.capex, hist.revenue, default=0.05))
    # Working capital cash flow item is -ΔNWC, typically. Keep ratio in [-0.2, 0.2]
    wc_pct = float(np.clip(_avg_ratio(hist.wc_item, hist.revenue, default=0.0), -0.2, 0.2))

    return ForecastDrivers(
        years=years,
        revenue_growth=revenue_growth,
        ebit_margin=ebit_margin,
        da_pct_rev=float(np.clip(da_pct, 0.0, 0.20)),
        capex_pct_rev=float(np.clip(capex_pct, 0.0, 0.30)),
        wc_item_pct_rev=wc_pct,
    )


def run_dcf(
    base_revenue: float,
    drivers: ForecastDrivers,
    tax_rate: float,
    wacc: float,
    terminal_growth: float,
    cash: float,
    debt: float,
    shares: float,
) -> DCFResult:
    if shares <= 0 or base_revenue <= 0 or wacc <= 0:
        return DCFResult(None, None, None, None, None)

    # guardrail
    g = float(terminal_growth)
    if g >= wacc - 0.01:
        g = wacc - 0.01

    rev = base_revenue
    pv_fcfs = 0.0
    fcfs = []

    for t in range(1, drivers.years + 1):
        rev = rev * (1.0 + drivers.revenue_growth[t - 1])
        ebit = rev * drivers.ebit_margin[t - 1]
        nopat = ebit * (1.0 - tax_rate)
        da = rev * drivers.da_pct_rev
        capex = rev * drivers.capex_pct_rev
        wc_item = rev * drivers.wc_item_pct_rev  # this is the cash flow line item (-ΔNWC)

        fcf = nopat + da - capex + wc_item
        fcfs.append(fcf)
        pv_fcfs += fcf / ((1.0 + wacc) ** t)

    terminal_fcf = fcfs[-1] * (1.0 + g)
    terminal_value = terminal_fcf / (wacc - g)
    pv_terminal = terminal_value / ((1.0 + wacc) ** drivers.years)

    ev = pv_fcfs + pv_terminal
    equity = ev - debt + cash
    vps = equity / shares

    return DCFResult(
        value_per_share=float(vps),
        enterprise_value=float(ev),
        equity_value=float(equity),
        pv_fcfs=float(pv_fcfs),
        pv_terminal=float(pv_terminal),
    )


def run_sensitivity(
    base_revenue: float,
    drivers: ForecastDrivers,
    tax_rate: float,
    base_wacc: float,
    base_g: float,
    cash: float,
    debt: float,
    shares: float,
    wacc_points: int = 5,
    g_points: int = 5,
) -> Tuple[List[float], List[float], List[List[Optional[float]]]]:
    waccs = list(np.linspace(max(0.03, base_wacc - 0.02), base_wacc + 0.02, wacc_points).astype(float))
    gs = list(np.linspace(max(-0.01, base_g - 0.01), base_g + 0.01, g_points).astype(float))

    matrix: List[List[Optional[float]]] = []
    for w in waccs:
        row = []
        for g in gs:
            res = run_dcf(
                base_revenue=base_revenue,
                drivers=drivers,
                tax_rate=tax_rate,
                wacc=float(w),
                terminal_growth=float(g),
                cash=cash,
                debt=debt,
                shares=shares,
            )
            row.append(res.value_per_share)
        matrix.append(row)

    return waccs, gs, matrix
