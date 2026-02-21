from __future__ import annotations

import os
from fastapi import FastAPI, Query, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .yahoo_provider import YahooProvider
from .valuation import compute_valuation

app = FastAPI(title="Valuation Backend", version="1.0")

# CORS: for private apps, you can lock this down later.
origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed = [o.strip() for o in origins.split(",") if o.strip()] if origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed if allowed != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

provider = YahooProvider()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/valuation")
def valuation(
    ticker: str = Query(..., description="Ticker symbol, e.g. AAPL or NOVO-B.CO"),
    use_llm: bool = Query(False, description="Enable LLM narrative + peer suggestions if OPENAI_API_KEY is set"),
    peers: str | None = Query(None, description="Optional comma-separated peer tickers"),
    years: int = Query(5, ge=3, le=10, description="Forecast years"),
    x_app_token: str | None = Header(None, description="Optional API token header for private deployments"),
):
    required = os.getenv("APP_TOKEN")
    if required:
        if not x_app_token or x_app_token != required:
            raise HTTPException(status_code=401, detail="Unauthorized")

    resp = compute_valuation(ticker=ticker, provider=provider, use_llm=use_llm, peers_csv=peers, years=years)
    return resp.model_dump()
