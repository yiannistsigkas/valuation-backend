from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PeerMetrics:
    ticker: str
    company_name: Optional[str]
    currency: Optional[str]
    market_cap: Optional[float]
    trailing_pe: Optional[float]
    ev_to_ebitda: Optional[float]
    ev_to_sales: Optional[float]


def _finite(xs: List[Optional[float]]) -> List[float]:
    out = []
    for v in xs:
        if v is None:
            continue
        try:
            fv = float(v)
            if np.isfinite(fv):
                out.append(fv)
        except Exception:
            continue
    return out


def summarize_multiples(peers: List[PeerMetrics]) -> Dict[str, Dict[str, Optional[float]]]:
    pe = _finite([p.trailing_pe for p in peers if p.trailing_pe and p.trailing_pe > 0])
    eve = _finite([p.ev_to_ebitda for p in peers if p.ev_to_ebitda and p.ev_to_ebitda > 0])
    evs = _finite([p.ev_to_sales for p in peers if p.ev_to_sales and p.ev_to_sales > 0])

    def stats(a: List[float]) -> Dict[str, Optional[float]]:
        if not a:
            return {"p25": None, "median": None, "p75": None}
        arr = np.array(a, dtype=float)
        return {
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }

    return {
        "P/E": stats(pe),
        "EV/EBITDA": stats(eve),
        "EV/Sales": stats(evs),
    }


def implied_prices(
    multiple_stats: Dict[str, Dict[str, Optional[float]]],
    target: Dict[str, Optional[float]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute implied price per share ranges for each multiple using P25/Median/P75."""

    shares = target.get("shares")
    if not shares or shares <= 0:
        return {}

    cash = float(target.get("cash") or 0.0)
    debt = float(target.get("debt") or 0.0)

    out: Dict[str, Dict[str, Optional[float]]] = {}

    # P/E: Equity = multiple * net_income
    net_income = target.get("net_income")
    if net_income and net_income > 0:
        pe_stats = multiple_stats.get("P/E", {})
        out["P/E"] = {
            "low": (pe_stats.get("p25") * net_income / shares) if pe_stats.get("p25") else None,
            "base": (pe_stats.get("median") * net_income / shares) if pe_stats.get("median") else None,
            "high": (pe_stats.get("p75") * net_income / shares) if pe_stats.get("p75") else None,
        }

    # EV/EBITDA: EV = multiple * ebitda -> Equity = EV - debt + cash
    ebitda = target.get("ebitda")
    if ebitda and ebitda > 0:
        m = multiple_stats.get("EV/EBITDA", {})
        def ev_to_price(mult: Optional[float]) -> Optional[float]:
            if not mult:
                return None
            ev = mult * ebitda
            eq = ev - debt + cash
            return eq / shares
        out["EV/EBITDA"] = {"low": ev_to_price(m.get("p25")), "base": ev_to_price(m.get("median")), "high": ev_to_price(m.get("p75"))}

    # EV/Sales: EV = multiple * revenue
    revenue = target.get("revenue")
    if revenue and revenue > 0:
        m = multiple_stats.get("EV/Sales", {})
        def evs_to_price(mult: Optional[float]) -> Optional[float]:
            if not mult:
                return None
            ev = mult * revenue
            eq = ev - debt + cash
            return eq / shares
        out["EV/Sales"] = {"low": evs_to_price(m.get("p25")), "base": evs_to_price(m.get("median")), "high": evs_to_price(m.get("p75"))}

    return out
