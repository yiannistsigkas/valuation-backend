# Valuation Backend (FastAPI + Yahoo Finance)

This is a ready-to-deploy backend API that powers a private valuation web app.

## What it does
- Fetches market & fundamentals using `yfinance` (Yahoo Finance)
- Runs a deterministic DCF (FCFF-style) + WACC
- Generates sensitivity (WACC × terminal growth)
- Optional peer-based relative valuation (P/E, EV/EBITDA, EV/Sales)
  - Provide peers manually via `?peers=`
  - Or enable LLM peer suggestions via `?use_llm=true` (requires `OPENAI_API_KEY`)
- Returns a single JSON payload from `GET /valuation`

## Endpoints
- `GET /health`
- `GET /valuation?ticker=AAPL&use_llm=false&years=5`
- `GET /valuation?ticker=AAPL&peers=MSFT,GOOGL,AMZN`

## Environment variables
- `APP_TOKEN` (optional) — if set, `/valuation` requires header `x-app-token: <APP_TOKEN>`
- `OPENAI_API_KEY` (optional)
- `OPENAI_MODEL` (optional, default: `gpt-4o-mini`)
- `ALLOWED_ORIGINS` (optional, comma-separated; default `*`)

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Test:
- http://localhost:8000/health
- http://localhost:8000/valuation?ticker=AAPL

## Deploy to Render (recommended)
Render's own guide shows a FastAPI deployment flow. You will create a Python Web Service and set a Start Command. 

Suggested start command (production style):
```bash
gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT
```

Build command:
```bash
pip install -r requirements.txt
```

After deployment, your base URL will look like:
`https://YOUR-SERVICE.onrender.com`

Use that as `BACKEND_BASE_URL` in Lovable.
