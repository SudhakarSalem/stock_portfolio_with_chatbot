"""
NSE Stock Portfolio Agent
=========================
FastAPI backend using:
  - yfinance          -> stock search suggestions + latest price/date (NSE)
  - google-genai SDK  -> Gemini LLM chat, restricted to portfolio questions
  - Pydantic v2       -> data validation
  - In-memory store   -> up to 10 holdings per session

Run:
    uvicorn main:app --reload
Then open http://localhost:8000
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from google import genai
from google.genai import types as genai_types

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("portfolio-agent")

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", extra="ignore")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    # gemini-3.5-flash does not exist as a released model. gemini-2.5-flash
    # is the current GA/stable Flash model; override via env var any time
    # (e.g. gemini-flash-latest, gemini-3-flash-preview) without touching code.
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    max_portfolio_stocks: int = Field(default=10, alias="MAX_PORTFOLIO_STOCKS")
    suggestion_cache_ttl: int = Field(default=300, alias="SUGGESTION_CACHE_TTL")
    price_cache_ttl: int = Field(default=60, alias="PRICE_CACHE_TTL")


settings = Settings()

genai_client: Optional[genai.Client] = None
if settings.gemini_api_key:
    genai_client = genai.Client(api_key=settings.gemini_api_key)
else:
    log.warning("GEMINI_API_KEY not set — chat endpoint will return an error until configured.")

# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(
    title="NSE Stock Portfolio Agent",
    description="Search NSE stocks, build a 10-stock portfolio, and chat with Gemini about it.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Simple TTL cache (no extra dependency needed)
# --------------------------------------------------------------------------


class TTLCache:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._store: dict[str, tuple[float, object]] = {}

    def get(self, key: str):
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: object):
        self._store[key] = (time.time() + self.ttl, value)


suggestion_cache = TTLCache(settings.suggestion_cache_ttl)
price_cache = TTLCache(settings.price_cache_ttl)

NSE_SUFFIX = ".NS"


# --------------------------------------------------------------------------
# Data models
# --------------------------------------------------------------------------


class StockSuggestion(BaseModel):
    symbol: str  # e.g. "TCS.NS"
    display_symbol: str  # e.g. "TCS"
    name: str
    exchange: str


class HoldingIn(BaseModel):
    symbol: str = Field(..., description="NSE symbol, e.g. TCS or TCS.NS")
    shares: float = Field(..., gt=0, description="Number of shares held")

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("Symbol cannot be empty")
        return v


class Holding(BaseModel):
    symbol: str
    display_symbol: str
    name: str
    shares: float
    price: float
    currency: str
    as_of: str  # ISO date/time of the quoted price
    value: float


class PortfolioResponse(BaseModel):
    holdings: list[Holding]
    total_value: float
    count: int
    capacity: int


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    allowed: bool
    timestamp: str


# --------------------------------------------------------------------------
# In-memory portfolio (single-session demo store)
# --------------------------------------------------------------------------

portfolio: dict[str, HoldingIn] = {}  # keyed by normalized display symbol (no .NS)


def to_yf_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    return symbol if symbol.endswith(NSE_SUFFIX) else f"{symbol}{NSE_SUFFIX}"


def to_display_symbol(symbol: str) -> str:
    return symbol[: -len(NSE_SUFFIX)] if symbol.upper().endswith(NSE_SUFFIX) else symbol.upper()


# --------------------------------------------------------------------------
# yfinance helpers
# --------------------------------------------------------------------------


def search_nse_suggestions(query: str, limit: int = 10) -> list[StockSuggestion]:
    """
    Autocomplete suggestions for NSE stocks using yfinance's built-in Search.
    (API Ninjas has no symbol-search endpoint, so yfinance's search — free,
    no API key — is used here instead.)
    """
    cache_key = f"sugg:{query.lower()}"
    cached = suggestion_cache.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    try:
        result = yf.Search(query, max_results=25, include_research=False)
        quotes = result.quotes or []
    except Exception as exc:
        log.error("yfinance search failed for %r: %s", query, exc)
        raise HTTPException(status_code=502, detail="Stock search service unavailable") from exc

    suggestions: list[StockSuggestion] = []
    for q in quotes:
        symbol = q.get("symbol", "")
        exch = (q.get("exchange") or "").upper()
        # Yahoo tags NSE listings as exchange "NSI" and always suffixes .NS
        if symbol.endswith(NSE_SUFFIX) or exch == "NSI":
            suggestions.append(
                StockSuggestion(
                    symbol=symbol if symbol.endswith(NSE_SUFFIX) else f"{symbol}{NSE_SUFFIX}",
                    display_symbol=to_display_symbol(symbol),
                    name=q.get("longname") or q.get("shortname") or symbol,
                    exchange="NSE",
                )
            )
        if len(suggestions) >= limit:
            break

    suggestion_cache.set(cache_key, suggestions)
    return suggestions


def get_latest_price(symbol: str) -> dict:
    """
    Latest close price + date for an NSE symbol via yfinance.
    """
    yf_symbol = to_yf_symbol(symbol)
    cache_key = f"price:{yf_symbol}"
    cached = price_cache.get(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    ticker = yf.Ticker(yf_symbol)

    try:
        hist = ticker.history(period="5d", interval="1d")
    except Exception as exc:
        log.error("yfinance history failed for %s: %s", yf_symbol, exc)
        raise HTTPException(status_code=502, detail="Price service unavailable") from exc

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No price data found for '{symbol}' on NSE")

    last_row = hist.iloc[-1]
    last_date = hist.index[-1]

    fast_info = {}
    try:
        fast_info = ticker.fast_info or {}
    except Exception:
        pass

    name = None
    try:
        info = ticker.get_info()
        name = info.get("longName") or info.get("shortName")
    except Exception:
        pass

    data = {
        "symbol": yf_symbol,
        "display_symbol": to_display_symbol(yf_symbol),
        "name": name or to_display_symbol(yf_symbol),
        "price": round(float(last_row["Close"]), 2),
        "currency": fast_info.get("currency", "INR"),
        "as_of": last_date.to_pydatetime().astimezone(timezone.utc).isoformat(),
    }
    price_cache.set(cache_key, data)
    return data


# --------------------------------------------------------------------------
# Domain restriction for the chat endpoint
# --------------------------------------------------------------------------

FINANCE_KEYWORDS = {
    "portfolio", "stock", "stocks", "share", "shares", "holding", "holdings",
    "price", "value", "worth", "gain", "loss", "profit", "return", "returns",
    "invest", "investment", "buy", "sell", "nse", "market", "equity", "asset",
    "diversif", "allocation", "dividend", "performance", "average", "cost",
    "risk", "rupee", "₹", "cagr", "capital", "wealth", "money", "fund",
}

OFF_TOPIC_HINTS = {
    "weather", "joke", "recipe", "movie", "song", "sports", "game", "news",
    "politic", "travel", "restaurant", "health advice", "relationship",
    "homework", "translate", "code review", "write a poem",
}


def is_finance_question(text: str) -> bool:
    t = text.lower()
    if any(h in t for h in OFF_TOPIC_HINTS):
        return False
    if any(k in t for k in FINANCE_KEYWORDS):
        return True
    # Fall back: mentions of a held symbol also count
    return any(sym.lower() in t for sym in portfolio.keys())


def build_portfolio_context() -> str:
    if not portfolio:
        return "The user's portfolio is currently empty."

    lines = ["Current portfolio holdings:"]
    total = 0.0
    for sym, holding in portfolio.items():
        try:
            quote = get_latest_price(sym)
        except HTTPException:
            continue
        value = quote["price"] * holding.shares
        total += value
        lines.append(
            f"- {sym}: {holding.shares} shares @ ₹{quote['price']} "
            f"(as of {quote['as_of']}) = ₹{value:.2f}"
        )
    lines.append(f"Total portfolio value: ₹{total:.2f}")
    return "\n".join(lines)


def ask_gemini(user_message: str) -> str:
    if genai_client is None:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    system_instruction = (
        "You are a financial assistant that ONLY discusses the user's stock "
        "portfolio: holdings, share counts, prices, valuations, gains/losses, "
        "diversification, and related NSE market questions. "
        "If asked anything outside personal-finance/portfolio topics, politely "
        "decline and redirect the user back to portfolio questions. "
        "Always reason using the portfolio context given below; use ₹ for currency.\n\n"
        f"{build_portfolio_context()}"
    )

    try:
        response = genai_client.models.generate_content(
            model=settings.gemini_model,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
                max_output_tokens=500,
            ),
        )
    except Exception as exc:
        log.error("Gemini call failed: %s", exc)
        raise HTTPException(status_code=502, detail="LLM service unavailable") from exc

    return (response.text or "").strip() or "I couldn't generate a response — please try again."


# --------------------------------------------------------------------------
# API endpoints
# --------------------------------------------------------------------------


@app.get("/api/health", tags=["Info"])
async def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/stocks/suggest", response_model=list[StockSuggestion], tags=["Stocks"])
async def suggest_stocks(q: str = Query(..., min_length=3, description="At least 3 letters")):
    """Autocomplete: type 3+ letters to get matching NSE stock names/symbols."""
    return search_nse_suggestions(q)


@app.get("/api/stocks/price/{symbol}", tags=["Stocks"])
async def stock_price(symbol: str):
    """Latest close price + date for a single NSE symbol."""
    return get_latest_price(symbol)


@app.get("/api/portfolio", response_model=PortfolioResponse, tags=["Portfolio"])
async def view_portfolio():
    holdings: list[Holding] = []
    total = 0.0
    for sym, h in portfolio.items():
        quote = get_latest_price(sym)
        value = round(quote["price"] * h.shares, 2)
        total += value
        holdings.append(
            Holding(
                symbol=quote["symbol"],
                display_symbol=quote["display_symbol"],
                name=quote["name"],
                shares=h.shares,
                price=quote["price"],
                currency=quote["currency"],
                as_of=quote["as_of"],
                value=value,
            )
        )
    return PortfolioResponse(
        holdings=holdings,
        total_value=round(total, 2),
        count=len(holdings),
        capacity=settings.max_portfolio_stocks,
    )


@app.post("/api/portfolio", response_model=PortfolioResponse, tags=["Portfolio"])
async def add_holding(holding: HoldingIn):
    display_symbol = to_display_symbol(holding.symbol)

    if display_symbol not in portfolio and len(portfolio) >= settings.max_portfolio_stocks:
        raise HTTPException(
            status_code=400,
            detail=f"Portfolio limit reached ({settings.max_portfolio_stocks} stocks max). "
            "Remove a stock before adding another.",
        )

    # Validate the symbol actually resolves to a price before accepting it
    get_latest_price(display_symbol)

    if display_symbol in portfolio:
        portfolio[display_symbol].shares += holding.shares
    else:
        portfolio[display_symbol] = HoldingIn(symbol=display_symbol, shares=holding.shares)

    return await view_portfolio()


@app.delete("/api/portfolio/{symbol}", response_model=PortfolioResponse, tags=["Portfolio"])
async def remove_holding(symbol: str):
    display_symbol = to_display_symbol(symbol)
    if display_symbol not in portfolio:
        raise HTTPException(status_code=404, detail=f"{display_symbol} is not in the portfolio")
    del portfolio[display_symbol]
    return await view_portfolio()


@app.delete("/api/portfolio", response_model=PortfolioResponse, tags=["Portfolio"])
async def clear_portfolio():
    portfolio.clear()
    return await view_portfolio()


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    if not is_finance_question(req.message):
        return ChatResponse(
            reply=(
                "I can only help with questions about your stock portfolio — "
                "holdings, prices, value, gains/losses, or NSE market topics. "
                "Please ask something about your portfolio."
            ),
            allowed=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    reply = ask_gemini(req.message)
    return ChatResponse(
        reply=reply,
        allowed=True,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------
# Frontend (static single-page app)
# --------------------------------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(__file__), "")
if os.path.isdir(STATIC_DIR):
    app.mount("", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
