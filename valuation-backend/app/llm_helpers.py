from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def _client():
    try:
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        return None


def has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def suggest_peers(
    company_name: str,
    ticker: str,
    sector: Optional[str],
    industry: Optional[str],
    country: Optional[str],
    max_peers: int = 12,
) -> List[str]:
    """Ask the LLM for a candidate peer list. Caller MUST validate tickers."""
    client = _client()
    if client is None:
        return []

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""You are helping build a comparable-company set for relative valuation.

Company:
- Name: {company_name}
- Ticker: {ticker}
- Sector: {sector or 'unknown'}
- Industry: {industry or 'unknown'}
- Country: {country or 'unknown'}

Task:
Return a JSON object with a single key "tickers" that is an array of {max_peers} or fewer publicly traded ticker symbols that would be reasonable peers.

Rules:
- Only include tickers (no company names).
- Prefer direct competitors and closest business models.
- Mix in 1-2 larger anchors if the peer set would be too small.
- If non-US listing suffix is needed (e.g., .ST, .L), include it.
- Do not include the target ticker.
- Output ONLY valid JSON.
"""

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            response_format={"type": "json_object"},
        )
        text = resp.output_text
        obj = json.loads(text)
        out = obj.get("tickers", [])
        if not isinstance(out, list):
            return []
        # normalize
        cleaned = []
        for x in out:
            if not isinstance(x, str):
                continue
            sym = x.strip().upper()
            if sym and sym != ticker.upper():
                cleaned.append(sym)
        # de-dup preserve order
        seen = set()
        uniq = []
        for s in cleaned:
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        return uniq[:max_peers]
    except Exception:
        return []


def make_narrative(context: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate short bullets for drivers/risks/notes. Uses LLM if available, otherwise rules."""
    if not has_openai():
        return _fallback_narrative(context)

    client = _client()
    if client is None:
        return _fallback_narrative(context)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Keep prompt small; provide only computed numbers and assumptions.
    prompt = {
        "role": "user",
        "content": (
            "Write concise valuation commentary based ONLY on the provided JSON context. "
            "Return JSON with keys drivers, risks, notes. Each is an array of short bullet strings. "
            "No numbers that are not in the context. 5-8 bullets for drivers, 4-6 for risks, 3-6 for notes.\n\n"
            + json.dumps(context)
        ),
    }

    try:
        resp = client.responses.create(
            model=model,
            input=[prompt],
            response_format={"type": "json_object"},
        )
        obj = json.loads(resp.output_text)
        out = {
            "drivers": obj.get("drivers", []),
            "risks": obj.get("risks", []),
            "notes": obj.get("notes", []),
        }
        # basic type safety
        for k in list(out.keys()):
            if not isinstance(out[k], list):
                out[k] = []
            out[k] = [str(x) for x in out[k] if isinstance(x, (str, int, float))]
        return out
    except Exception:
        return _fallback_narrative(context)


def _fallback_narrative(context: Dict[str, Any]) -> Dict[str, List[str]]:
    meta = context.get("meta", {})
    asmp = context.get("assumptions", {})
    dcf = context.get("dcf", {})

    drivers = []
    if asmp.get("revenueGrowth"):
        drivers.append("Revenue growth path drives most of the forecast value.")
    drivers.append("Terminal value is a large share of the DCF; long-run assumptions matter.")
    drivers.append("WACC (discount rate) is a key driver of the present value.")

    risks = [
        "Financial statement line items from Yahoo Finance can be missing or inconsistent across tickers.",
        "Sensitivity results can vary materially with small changes in WACC or terminal growth.",
        "Comparable multiples may be distorted if peers have different profitability or leverage profiles.",
    ]

    notes = [
        "All calculations are deterministic; narrative text may be LLM-generated if enabled.",
        "Review and override assumptions if they do not match your investment case.",
    ]

    return {"drivers": drivers, "risks": risks, "notes": notes}
