from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import (
    Assumptions,
    CompsSummary,
    DataQualityIssue,
    DCFBridge,
    DCFOutput,
    FootballBar,
    Market,
    Meta,
    Narrative,
    Peer,
    Sensitivity,
    ValuationResponse,
)
from .yahoo_provider import YahooProvider, extract_statement_series
from .dcf_engine import build_historical, infer_drivers, run_dcf, run_sensitivity
from .comps_engine import PeerMetrics, summarize_multiples, implied_prices
from .llm_helpers import has_openai, make_narrative, suggest_peers


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        fx = float(x)
        if not np.isfinite(fx):
            return None
        return fx
    except Exception:
        return None


def _latest_from_series(s) -> Optional[float]:
    if s is None:
        return None
    try:
        # s index are periods; sort
        idx = list(s.index)
        try:
            idx_sorted = sorted(idx, reverse=True)
        except Exception:
            idx_sorted = idx
        for p in idx_sorted:
            v = _safe_float(s.get(p))
            if v is not None:
                return v
    except Exception:
        return None
    return None


def compute_valuation(
    ticker: str,
    provider: YahooProvider,
    use_llm: bool = False,
    peers_csv: Optional[str] = None,
    years: int = 5,
) -> ValuationResponse:
    issues: List[DataQualityIssue] = []

    data = provider.fetch_company_data(ticker)
    info = data.info or {}

    company_name = info.get("longName") or info.get("shortName")
    currency = info.get("currency")
    country = info.get("country")
    sector = info.get("sector")
    industry = info.get("industry")

    price = _safe_float(info.get("regularMarketPrice"))
    shares = _safe_float(info.get("sharesOutstanding")) or _safe_float(info.get("impliedSharesOutstanding"))
    market_cap = _safe_float(info.get("marketCap"))
    beta = _safe_float(info.get("beta"))

    if market_cap is None and price is not None and shares is not None:
        market_cap = price * shares

    # Statements
    series = extract_statement_series(data)
    hist = build_historical(series)

    # Fallbacks
    base_revenue = None
    if hist.revenue:
        base_revenue = next((v for v in hist.revenue if v is not None), None)
    if base_revenue is None:
        base_revenue = _safe_float(info.get("totalRevenue"))

    if base_revenue is None:
        issues.append(DataQualityIssue(level="error", message="Missing revenue data; cannot run DCF."))
        base_revenue = 0.0

    # Tax rate
    tax_rate = hist.tax_rate
    if tax_rate is None:
        tax_rate = 0.21
        issues.append(DataQualityIssue(level="warning", message="Tax rate not found; using default 21%."))

    # Cash and debt
    cash = _safe_float(info.get("totalCash"))
    debt = _safe_float(info.get("totalDebt"))

    if cash is None:
        cash = _latest_from_series(series.get("cash"))
    if debt is None:
        debt = _latest_from_series(series.get("debt"))

    cash = float(cash or 0.0)
    debt = float(debt or 0.0)

    if shares is None or shares <= 0:
        issues.append(DataQualityIssue(level="error", message="Shares outstanding missing; cannot compute per-share value."))
        shares = 0.0

    # WACC inputs (simple defaults; user can override later)
    risk_free = 0.04
    erp = 0.05
    beta_used = float(beta if beta is not None else 1.0)
    cost_of_equity = risk_free + beta_used * erp

    # crude cost of debt fallback
    cost_of_debt = 0.05

    # weights
    equity_val = float(market_cap or 0.0)
    total_cap = equity_val + debt
    if total_cap <= 0:
        w_e = 1.0
        w_d = 0.0
    else:
        w_e = equity_val / total_cap
        w_d = debt / total_cap

    wacc = w_e * cost_of_equity + w_d * cost_of_debt * (1.0 - tax_rate)
    wacc = float(np.clip(wacc, 0.05, 0.15))

    terminal_growth = 0.025

    drivers = infer_drivers(hist, years=years, terminal_growth=terminal_growth)

    # Run DCF
    dcf_res = run_dcf(
        base_revenue=float(base_revenue),
        drivers=drivers,
        tax_rate=float(tax_rate),
        wacc=float(wacc),
        terminal_growth=float(terminal_growth),
        cash=float(cash),
        debt=float(debt),
        shares=float(shares),
    )

    # Sensitivity
    waccs, gs, matrix = run_sensitivity(
        base_revenue=float(base_revenue),
        drivers=drivers,
        tax_rate=float(tax_rate),
        base_wacc=float(wacc),
        base_g=float(terminal_growth),
        cash=float(cash),
        debt=float(debt),
        shares=float(shares),
    )

    # Comps
    peer_tickers: List[str] = []
    if peers_csv:
        peer_tickers = [p.strip().upper() for p in peers_csv.split(",") if p.strip()]
    elif use_llm and has_openai() and company_name:
        peer_tickers = suggest_peers(company_name, ticker, sector, industry, country)

    peer_objs: List[Peer] = []
    peer_metrics: List[PeerMetrics] = []

    # Target metrics for implied prices
    target_net_income = _safe_float(info.get("netIncomeToCommon"))
    target_ebitda = _safe_float(info.get("ebitda"))
    target_revenue = _safe_float(info.get("totalRevenue")) or float(base_revenue)

    for pt in peer_tickers[:15]:
        try:
            pdta = provider.fetch_company_data(pt)
            pinf = pdta.info or {}
            # Basic validation: same sector if available
            if sector and pinf.get("sector") and pinf.get("sector") != sector:
                continue

            peer = Peer(
                ticker=pt,
                companyName=pinf.get("longName") or pinf.get("shortName"),
                marketCap=_safe_float(pinf.get("marketCap")),
                currency=pinf.get("currency"),
                trailingPE=_safe_float(pinf.get("trailingPE")),
                evToEbitda=_safe_float(pinf.get("enterpriseToEbitda")),
                evToSales=_safe_float(pinf.get("enterpriseToRevenue")),
            )
            peer_objs.append(peer)
            peer_metrics.append(
                PeerMetrics(
                    ticker=pt,
                    company_name=peer.companyName,
                    currency=peer.currency,
                    market_cap=peer.marketCap,
                    trailing_pe=peer.trailingPE,
                    ev_to_ebitda=peer.evToEbitda,
                    ev_to_sales=peer.evToSales,
                )
            )
        except Exception:
            continue

    multiples = summarize_multiples(peer_metrics)
    implied = implied_prices(
        multiples,
        {
            "shares": shares,
            "cash": cash,
            "debt": debt,
            "net_income": target_net_income,
            "ebitda": target_ebitda,
            "revenue": target_revenue,
        },
    )

    comps_summary = CompsSummary(
        peers=peer_objs,
        multiples=multiples,
        impliedPriceRanges=implied,
    )

    if not peer_tickers:
        issues.append(DataQualityIssue(level="info", message="No peer set was generated. Provide peers manually or enable LLM peers."))

    # Football field
    football: List[FootballBar] = []

    # DCF sensitivity range
    flat = [v for row in matrix for v in row if v is not None]
    if flat:
        football.append(
            FootballBar(
                method="DCF sensitivity",
                low=float(np.min(flat)),
                base=dcf_res.value_per_share,
                high=float(np.max(flat)),
            )
        )
    else:
        football.append(FootballBar(method="DCF sensitivity", low=None, base=dcf_res.value_per_share, high=None))

    # Multiples ranges
    for k, r in implied.items():
        football.append(FootballBar(method=f"{k} comps", low=r.get("low"), base=r.get("base"), high=r.get("high")))

    # Narrative
    narrative_ctx = {
        "meta": {"ticker": ticker, "companyName": company_name, "currency": currency},
        "assumptions": {
            "years": years,
            "wacc": wacc,
            "terminalGrowth": terminal_growth,
            "taxRate": tax_rate,
            "revenueGrowth": drivers.revenue_growth,
            "ebitMargin": drivers.ebit_margin,
            "daPctRevenue": drivers.da_pct_rev,
            "capexPctRevenue": drivers.capex_pct_rev,
            "wcItemPctRevenue": drivers.wc_item_pct_rev,
        },
        "dcf": {
            "valuePerShare": dcf_res.value_per_share,
            "enterpriseValue": dcf_res.enterprise_value,
        },
        "market": {"price": price, "marketCap": market_cap, "beta": beta_used},
        "comps": {"peerCount": len(peer_objs), "multiples": multiples},
        "dataQuality": [i.message for i in issues],
    }

    if use_llm:
        narr = make_narrative(narrative_ctx)
        narrative = Narrative(drivers=narr.get("drivers", []), risks=narr.get("risks", []), notes=narr.get("notes", []))
    else:
        narr = make_narrative(narrative_ctx)  # fallback even when disabled
        narrative = Narrative(drivers=narr.get("drivers", []), risks=narr.get("risks", []), notes=narr.get("notes", []))

    # If LLM disabled, add a note
    if not use_llm:
        issues.append(DataQualityIssue(level="info", message="LLM features disabled; using rule-based narrative and no automatic peers."))

    assumptions = Assumptions(
        years=years,
        baseYear=hist.periods[0] if hist.periods else None,
        terminalGrowth=float(terminal_growth),
        wacc=float(wacc),
        riskFreeRate=float(risk_free),
        equityRiskPremium=float(erp),
        costOfDebt=float(cost_of_debt),
        taxRate=float(tax_rate),
        revenueGrowth=[float(x) for x in drivers.revenue_growth],
        ebitMargin=[float(x) for x in drivers.ebit_margin],
        daPctRevenue=float(drivers.da_pct_rev),
        capexPctRevenue=float(drivers.capex_pct_rev),
        wcItemPctRevenue=float(drivers.wc_item_pct_rev),
    )

    dcf_out = DCFOutput(
        valuePerShare=dcf_res.value_per_share,
        pvOfFcfs=dcf_res.pv_fcfs,
        pvOfTerminal=dcf_res.pv_terminal,
        bridge=DCFBridge(
            enterpriseValue=dcf_res.enterprise_value,
            cash=cash,
            totalDebt=debt,
            equityValue=dcf_res.equity_value,
        ),
    )

    resp = ValuationResponse(
        meta=Meta(
            ticker=ticker.upper(),
            companyName=company_name,
            currency=currency,
            asOf=str(date.today()),
        ),
        market=Market(price=price, shares=shares or None, marketCap=market_cap, beta=beta),
        assumptions=assumptions,
        dcf=dcf_out,
        sensitivity=Sensitivity(wacc=[float(x) for x in waccs], g=[float(x) for x in gs], valuePerShareMatrix=matrix),
        comps=comps_summary,
        football=football,
        narrative=narrative,
        dataQuality=issues,
        raw={
            "sector": sector,
            "industry": industry,
            "country": country,
        },
    )

    return resp
