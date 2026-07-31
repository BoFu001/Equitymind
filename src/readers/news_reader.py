"""
src/readers/news_reader.py

News Data Acquisition — fetches and filters company news, with no
sentiment scoring. This is a pure data-layer module: it retrieves raw
articles from finlight.me and filters out articles that aren't actually
about the target company, then returns a clean article list.

Sentiment scoring (FinBERT) is deliberately NOT done here — that is a
compute-layer concern and lives in src/quant/news_sentiment_signal.py,
consistent with the project's data-layer / compute-layer separation:
this module does I/O and filtering only, never computation.

Filtering approach — company "common name" matching, not full legal
name or ticker matching:
    A naive approach (match on ticker, or on the full legal company_name)
    was tried first and found to have two real problems, verified against
    real finlight news data for 8+ companies (2026-07-18):
      1. Ticker-only matching (e.g. "TSLA") lets through leveraged/
         options-income ETF noise — products like "YieldMax TSLA Option
         Income Strategy ETF" use the ticker in their own product name,
         so they pass a ticker filter even though they aren't news about
         the company itself.
      2. Full legal name matching (e.g. "Tesla, Inc.") fails because
         news headlines almost never use the full legal name — they use
         a short common name ("Tesla") instead. String-splitting rules to
         extract this common name from the legal name (first word, split
         on comma, strip punctuation) were tried and repeatedly broke on
         real edge cases (e.g. "The Coca-Cola Company" -> "The" via naive
         first-word split; "Amazon.com, Inc." -> "Amazoncom" via naive
         punctuation stripping).
    The fix: match on a company's "common_name" — the short name a
    headline would actually use — precomputed via LLM for the whole
    stock universe (see scripts/update_common_names.py) and stored in
    stock_universe.json. Matching on common_name alone (not ticker)
    naturally avoids the ETF noise from problem 1, since those products
    are named after the ticker, not the common name.
"""

import json
from pathlib import Path

import requests
import yfinance as yf
from datetime import datetime, timedelta
from openai import OpenAI

from config import FINLIGHT_API_KEY, OPENAI_API_KEY, LLM_MODEL_LIGHT

client = OpenAI(api_key=OPENAI_API_KEY)

UNIVERSE_PATH = Path(__file__).parent.parent / "quant" / "data" / "stock_universe.json"


# ─────────────────────────────────────────────
# Common name lookup (precomputed, with LLM fallback)
# ─────────────────────────────────────────────

def _get_common_name(ticker: str) -> str:
    """
    Returns the short, common name a news headline would use for this
    company (e.g. "Tesla" for "Tesla, Inc.").

    Looks up stock_universe.json first, since common_name is precomputed
    there for the whole 250-ticker universe (see
    scripts/update_common_names.py) — this is the fast, free path for
    any ticker already in the universe, and covers the vast majority of
    real traffic.

    Falls back to a live yfinance lookup + LLM call only for tickers NOT
    in the universe (e.g. small-caps, recent IPOs) — this result is not
    cached, since it's expected to be an occasional exception, not the
    common case. This module resolves the full legal name itself in that
    fallback case (rather than requiring the caller to supply it), so
    that news fetching has no dependency on any other data source
    (e.g. fetch_data's stock_snapshot) and can run fully in parallel
    with it.
    """
    try:
        with open(UNIVERSE_PATH, "r") as f:
            universe = json.load(f)
        details = universe.get("details", {})
        if ticker in details and "common_name" in details[ticker]:
            print(f"  [_get_common_name] {ticker}: UNIVERSE HIT — {details[ticker]['common_name']!r}")
            return details[ticker]["common_name"]
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback: ticker not in the precomputed universe. Resolve the full
    # legal name ourselves (one lightweight yfinance call) rather than
    # depending on a snapshot fetched elsewhere, then ask the LLM live.
    print(f"  [_get_common_name] {ticker}: FALLBACK PATH — not in universe, querying yfinance...")
    try:
        company_name = yf.Ticker(ticker).info.get("longName") or ticker
    except Exception:
        company_name = ticker
    print(f"  [_get_common_name] {ticker}: FALLBACK PATH — yfinance longName={company_name!r}")

    prompt = f'''The company\'s official full legal name is: "{company_name}"

Extract the short, common name this company is most often called in
English financial news headlines (not the ticker symbol — the common
name people actually use, in English).

Examples:
"Microsoft Corporation" -> "Microsoft"
"Tesla, Inc." -> "Tesla"
"Amazon.com, Inc." -> "Amazon"
"The Coca-Cola Company" -> "Coca-Cola"
"GE Aerospace" -> "GE"

Reply with ONLY the common name itself, no other text.'''

    response = client.chat.completions.create(
        model=LLM_MODEL_LIGHT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Relevance filtering
# ─────────────────────────────────────────────

def _title_mentions_company(title: str, common_name: str) -> bool:
    """
    Returns True if the company's common name appears in the article's
    title itself — not just somewhere in the body. This excludes
    broader industry/market news that only mentions the company in
    passing, and (per the module docstring above) avoids leveraged ETF
    product noise that a ticker-based match would let through.
    """
    return common_name.lower() in title.lower()


# ─────────────────────────────────────────────
# Subscription tier check (defensive)
# ─────────────────────────────────────────────

def _warn_if_downgraded(articles: list) -> None:
    """
    Defensive check: the 'companies' field is only returned on paid tiers
    (Pro Light and above) — verified empirically, since finlight docs
    don't specify exactly which tier unlocks it. If we requested
    includeEntities but got no 'companies' field on ANY article, this is
    a strong signal that the subscription has reverted to the free
    Launchpad tier (e.g. payment issue), so we print a clear warning
    rather than silently degrading.
    """
    if not articles:
        return
    has_companies_field = any(a.get("companies") is not None for a in articles)
    if not has_companies_field:
        print(
            "  [news_reader] WARNING: no 'companies' field in "
            "response — this suggests the finlight subscription may have "
            "reverted to the free tier (e.g. payment issue). Check your "
            "billing at https://app.finlight.me/"
        )

# ─────────────────────────────────────────────
# Main entry point: fetch + filter
# ─────────────────────────────────────────────

def fetch_company_news(ticker: str, max_articles: int = 100, days_back: int = 30) -> list:
    """
    Fetches recent news for a ticker from finlight.me, then filters to
    only company-specific articles (title contains the company's common
    name — excludes broader industry/market news, and excludes
    leveraged/options ETF products named after the ticker).

    No sentiment scoring happens here — see module docstring. Returns a
    list of already-filtered articles with basic fields only.

    Takes only a ticker — no company_name input. The common name is
    resolved entirely inside this module (precomputed universe lookup,
    with a self-contained yfinance + LLM fallback for tickers outside
    it — see _get_common_name), so this function has no dependency on
    any other data source and can be fetched fully in parallel with
    stock_snapshot, risk_inputs, quality_inputs, consensus_inputs, etc.

    Args:
        ticker: stock ticker symbol, e.g. "TSLA"
        max_articles: max articles to request from finlight (1-100)
        days_back: how many days back to search

    Returns:
        list of dicts, each with "title", "url", "published", "summary" —
        already filtered to company-specific articles only.
    """
    common_name = _get_common_name(ticker)

    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        url = "https://api.finlight.me/v2/articles"
        headers = {
            "X-API-KEY":    FINLIGHT_API_KEY,
            "Content-Type": "application/json",
        }
        body = {
            "query":          ticker,
            "tickers":        [ticker],
            "language":       "en",
            "from":           date_from,
            "pageSize":       max_articles,
            "includeEntities": True,
        }
        response = requests.post(url, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        raw_articles = response.json().get("articles", [])
        _warn_if_downgraded(raw_articles)

    except Exception as e:
        print(f"  [news_reader] Error fetching news for {ticker}: {e}")
        return []

    filtered = [
        a for a in raw_articles
        if _title_mentions_company(a.get("title", ""), common_name)
    ]

    results = [
        {
            "title":     a.get("title", ""),
            "url":       a.get("link", ""),
            "published": a.get("publishDate", ""),
            "summary":   a.get("summary", "") or "",
        }
        for a in filtered
    ]

    print(
        f"  [news_reader] {ticker}: {len(raw_articles)} fetched, "
        f"{len(results)} relevant (common_name={common_name!r})"
    )
    return results
