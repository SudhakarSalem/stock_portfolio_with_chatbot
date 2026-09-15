# NSE Portfolio Agent (FastAPI + yfinance + Gemini)

A single FastAPI app that:

- **Searches NSE stocks** as you type (3+ letters) and **fetches the latest
  price + date** — both via `yfinance` (free, no API key, no rate-limit key
  management).
- Lets you build a **portfolio of up to 10 stocks** with share counts, and
  shows **total value** + per-stock value.
- Has a **Gemini-powered chat** that answers *only* portfolio/financial
  questions — anything else (jokes, weather, general chit-chat, etc.) is
  politely declined before it ever reaches the model.

## ⚠️ Two things worth knowing before you run this

1. **"API Ninjas for stock name suggestions" isn't actually possible.**
   API Ninjas' Stock Price API only accepts a ticker you already know — it
   has no name/symbol search endpoint. So autocomplete here uses
   **`yfinance.Search()`**, which is real, free, and covers NSE (`.NS`)
   listings. Nothing fake was wired up in its place.
2. **`gemini-3.5-flash` doesn't exist** as a released Gemini model. The app
   defaults to **`gemini-2.5-flash`** (current GA/stable Flash model) and
   reads the model name from `GEMINI_MODEL` in `.env`, so you can point it
   at `gemini-flash-latest` or `gemini-3-flash-preview` the moment you want
   to, with zero code changes.

## Stack (latest stable, verified on PyPI)

| Package | Version |
|---|---|
| fastapi | 0.141.1 |
| uvicorn[standard] | 0.52.4 |
| pydantic / pydantic-settings | 2.13.5 / 2.15.0 |
| yfinance | 0.2.55 |
| google-genai | 1.28.0 |

`google-genai` is Google's **current** SDK (`from google import genai`).
The older `google-generativeai` package is archived/deprecated — it doesn't
get new Gemini features, so it isn't used here.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your key from https://aistudio.google.com/apikey

uvicorn main:app --reload
```

Open **http://localhost:8000** — the UI is served directly by FastAPI
(`static/index.html`), no separate frontend server needed.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/stocks/suggest?q=TCS` | Autocomplete (min 3 chars) |
| GET | `/api/stocks/price/{symbol}` | Latest price + date for one symbol |
| GET | `/api/portfolio` | View holdings + total value |
| POST | `/api/portfolio` | `{"symbol": "TCS", "shares": 10}` — add/update (max 10 stocks) |
| DELETE | `/api/portfolio/{symbol}` | Remove one holding |
| DELETE | `/api/portfolio` | Clear portfolio |
| POST | `/api/chat` | `{"message": "..."}` — Gemini, portfolio-only |

Interactive docs: **http://localhost:8000/docs**

## Notes on yfinance for NSE

- Symbols are auto-suffixed with `.NS` (e.g. `TCS` → `TCS.NS`) — Yahoo's
  convention for NSE-listed equities.
- Price + date come from `Ticker.history(period="5d")`, taking the most
  recent trading day's close (handles weekends/holidays automatically).
- Both search and price responses are cached briefly in memory
  (`SUGGESTION_CACHE_TTL` / `PRICE_CACHE_TTL` in `.env`) to avoid hammering
  Yahoo's endpoints — yfinance scrapes public endpoints and has no official
  SLA or key-based rate limit, so be reasonable with request volume.

## Production notes

This keeps the portfolio in memory for simplicity (one shared session,
resets on restart). Before shipping this for real users, add a proper
per-user store (DB) and authentication — that part was intentionally left
out to keep the example focused.
