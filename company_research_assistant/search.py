"""Search tool abstractions for company research."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass(frozen=True)
class StockQuote:
    symbol: str
    price: float | None
    currency: str | None = None
    change: float | None = None
    change_percent: float | None = None
    market_time: str | None = None
    source: str = "unavailable"
    error: str | None = None


class CompanySearchTool(Protocol):
    """Protocol implemented by concrete search providers."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Return search results for a company research query."""


class StockQuoteTool(Protocol):
    """Protocol implemented by quote providers."""

    def quote(self, symbol: str) -> StockQuote:
        """Return a current stock quote for a ticker symbol."""


class TavilyCompanySearchTool:
    """Tavily-backed search tool.

    Tavily MCP can be wired behind this same protocol in production. For a
    standalone Python app, the official Tavily client keeps the example simple.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Tavily API key is required.")
        self.api_key = api_key

    @classmethod
    def from_env(cls) -> "TavilyCompanySearchTool":
        return cls(os.environ["TAVILY_API_KEY"])

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise RuntimeError(
                "Install tavily-python or use a different CompanySearchTool."
            ) from exc

        client = TavilyClient(api_key=self.api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
        )
        return [
            SearchResult(
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=float(item.get("score") or 0.0),
            )
            for item in response.get("results", [])
        ]


class LocalCompanySearchTool:
    """Deterministic offline fallback for tests and demos without API keys."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        company = query.split(" company research:", 1)[-1].split(" | ", 1)[0].strip()
        if not company:
            company = query.split()[0] if query.split() else "the company"

        snippets = [
            SearchResult(
                title=f"{company} recent developments",
                url="https://example.com/recent-developments",
                content=(
                    f"{company} has recently focused on product launches, market "
                    "positioning, operational execution, and customer growth."
                ),
                score=0.82,
            ),
            SearchResult(
                title=f"{company} financial overview",
                url="https://example.com/financials",
                content=(
                    f"Recent coverage of {company} discusses revenue trends, "
                    "profitability, investment priorities, and investor sentiment."
                ),
                score=0.78,
            ),
            SearchResult(
                title=f"{company} competitive landscape",
                url="https://example.com/competitors",
                content=(
                    f"{company} competes with established peers and emerging "
                    "challengers across its main markets."
                ),
                score=0.73,
            ),
        ]
        return snippets[:max_results]


class YahooStockQuoteTool:
    """Fetch quote data with a Yahoo-first, Stooq-fallback strategy."""

    def quote(self, symbol: str) -> StockQuote:
        try:
            import requests
        except ImportError as exc:
            return StockQuote(
                symbol=symbol.upper(),
                price=None,
                source="Yahoo Finance",
                error=f"requests is not installed: {exc}",
            )

        normalized = symbol.upper().strip()
        yahoo_quote = self._quote_from_yahoo(requests, normalized)
        if yahoo_quote.price is not None:
            return yahoo_quote

        stooq_quote = self._quote_from_stooq(requests, normalized)
        if stooq_quote.price is not None:
            return stooq_quote

        fallback_error = yahoo_quote.error or stooq_quote.error
        if yahoo_quote.error and stooq_quote.error:
            fallback_error = f"Yahoo Finance: {yahoo_quote.error}; Stooq: {stooq_quote.error}"
        return StockQuote(
            symbol=normalized,
            price=None,
            source="Yahoo Finance and Stooq",
            error=fallback_error,
        )

    def _quote_from_yahoo(self, requests_module, symbol: str) -> StockQuote:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        try:
            response = requests_module.get(
                url,
                params={"range": "1d", "interval": "1m"},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    ),
                    "Accept": "application/json,text/plain,*/*",
                },
                timeout=8,
            )
            response.raise_for_status()
            payload = response.json()
            result = payload.get("chart", {}).get("result", [])
            if not result:
                return StockQuote(
                    symbol=symbol,
                    price=None,
                    source="Yahoo Finance",
                    error="No quote data returned.",
                )

            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice")
            previous_close = meta.get("previousClose")
            change = None
            change_percent = None
            if price is not None and previous_close:
                change = float(price) - float(previous_close)
                change_percent = (change / float(previous_close)) * 100

            market_timestamp = meta.get("regularMarketTime")
            market_time = None
            if market_timestamp:
                market_time = datetime.fromtimestamp(
                    int(market_timestamp),
                    tz=timezone.utc,
                ).isoformat()

            return StockQuote(
                symbol=symbol,
                price=float(price) if price is not None else None,
                currency=meta.get("currency"),
                change=change,
                change_percent=change_percent,
                market_time=market_time,
                source="Yahoo Finance",
            )
        except Exception as exc:
            return StockQuote(
                symbol=symbol,
                price=None,
                source="Yahoo Finance",
                error=str(exc),
            )

    def _quote_from_stooq(self, requests_module, symbol: str) -> StockQuote:
        stooq_symbol = _to_stooq_symbol(symbol)
        url = "https://stooq.com/q/l/"
        try:
            response = requests_module.get(
                url,
                params={"s": stooq_symbol, "f": "sd2t2ohlcv", "h": "", "e": "csv"},
                headers={"User-Agent": "company-research-assistant/0.1"},
                timeout=8,
            )
            response.raise_for_status()
            lines = [line.strip() for line in response.text.splitlines() if line.strip()]
            if len(lines) < 2:
                return StockQuote(
                    symbol=symbol,
                    price=None,
                    source="Stooq",
                    error="No CSV quote row returned.",
                )

            header = [item.strip().lower() for item in lines[0].split(",")]
            row = [item.strip() for item in lines[1].split(",")]
            values = dict(zip(header, row))
            close = values.get("close")
            if not close or close.upper() == "N/D":
                return StockQuote(
                    symbol=symbol,
                    price=None,
                    source="Stooq",
                    error="No close price returned.",
                )

            market_time = None
            date = values.get("date")
            time = values.get("time")
            if date and time and date.upper() != "N/D" and time.upper() != "N/D":
                market_time = f"{date}T{time}"

            return StockQuote(
                symbol=symbol,
                price=float(close),
                currency="USD" if stooq_symbol.endswith(".us") else None,
                market_time=market_time,
                source="Stooq",
            )
        except Exception as exc:
            return StockQuote(
                symbol=symbol,
                price=None,
                source="Stooq",
                error=str(exc),
            )


class LocalStockQuoteTool:
    """Offline quote fallback used when live quote lookup is unavailable."""

    def quote(self, symbol: str) -> StockQuote:
        return StockQuote(
            symbol=symbol.upper().strip(),
            price=None,
            source="local fallback",
            error="Live stock quote lookup is not configured or unavailable.",
        )


def default_search_tool() -> CompanySearchTool:
    if os.environ.get("TAVILY_API_KEY"):
        return TavilyCompanySearchTool.from_env()
    return LocalCompanySearchTool()


def default_stock_quote_tool() -> StockQuoteTool:
    return YahooStockQuoteTool()


def _to_stooq_symbol(symbol: str) -> str:
    normalized = symbol.lower().replace("-", ".")
    if "." in normalized:
        return normalized
    return f"{normalized}.us"
