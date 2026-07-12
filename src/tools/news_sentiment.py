import requests
from datetime import datetime, timedelta
from transformers import pipeline
from config import FINLIGHT_API_KEY

# Load FinBERT sentiment model once at module level
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
)


def get_news_and_sentiment(ticker: str, max_articles: int = 100, days_back: int = 30) -> list:
    """
    Fetches recent news for a ticker from finlight.me v2 API.
    Scores sentiment using FinBERT (financial domain model).
    Returns a list of articles with sentiment scores and URLs.

    NOTE: Currently using Finlight Pro Light plan ($29/month) — provides
    real-time access (no delay) and company entity data. Sentiment
    scoring itself is done locally via FinBERT, not finlight's own
    sentiment feature (which requires the Pro Standard tier and isn't
    used here). See _warn_if_downgraded() for a runtime check that
    detects if the subscription ever reverts to the free tier.
    """
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
            "includeEntities": True,  # only used to detect subscription
                                       # tier status (see _warn_if_downgraded)
                                       # — NOT used for relevance filtering,
                                       # since testing showed the confidence
                                       # score doesn't distinguish relevant
                                       # from irrelevant articles
        }
        response = requests.post(url, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        _warn_if_downgraded(articles)

    except Exception as e:
        print(f"  [news_sentiment] Error fetching news: {e}")
        return []

    results = []
    for article in articles:
        title    = article.get("title", "")
        summary  = article.get("summary", "") or ""
        link     = article.get("link", "")
        published = article.get("publishDate", "")

        # Score sentiment using FinBERT
        text = f"{title}. {summary}"[:512]
        try:
            result = finbert(text)[0]
            label  = result["label"]
            score  = round(result["score"], 4)
        except Exception:
            label = "neutral"
            score = 0.0

        results.append({
            "title":     title,
            "url":       link,
            "published": published,
            "summary":   summary,
            "sentiment": label,
            "score":     score,
        })

    print(f"  [news_sentiment] {len(results)} articles fetched for {ticker} (last {days_back} days)")
    return results

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
            "  [news_sentiment] \u26a0\ufe0f WARNING: no 'companies' field in "
            "response — this suggests the finlight subscription may have "
            "reverted to the free tier (e.g. payment issue). Check your "
            "billing at https://app.finlight.me/"
        )


def _is_company_specific(title: str, ticker: str, company_name: str) -> bool:
    """
    Returns True if the ticker or company name appears in the article's
    title itself — not just somewhere in the body. This is a simple
    heuristic to exclude broader industry/market news that only
    mentions the company in passing (e.g. "PC market crisis" articles
    that happen to name-drop a company). Verified against real AAPL,
    MU, CLOV, and COMP news samples during development — no meaningful
    false negatives found.
    """
    title_lower = title.lower()
    return ticker.lower() in title_lower or company_name.lower() in title_lower


def _aggregate_sentiment(articles: list) -> dict:
    """
    Aggregates a list of already-filtered, company-specific articles
    into a single net sentiment score and counts.

    net_score is the average of each article's signed confidence score
    (positive articles contribute +score, negative articles contribute
    -score, neutral articles contribute 0), clipped to [-1, 1] for
    consistency with every other signal engine in this system.
    """
    total = len(articles)

    positive = [a for a in articles if a["sentiment"] == "positive"]
    negative = [a for a in articles if a["sentiment"] == "negative"]
    neutral  = [a for a in articles if a["sentiment"] == "neutral"]

    net_score = sum(
        a["score"] if a["sentiment"] == "positive" else
        -a["score"] if a["sentiment"] == "negative" else
        0
        for a in articles
    ) / total

    net_score = round(max(-1.0, min(1.0, net_score)), 4)

    return {
        "net_score":      net_score,
        "positive_count": len(positive),
        "negative_count": len(negative),
        "neutral_count":  len(neutral),
        "total_articles": total,
    }


def _sentiment_label(net_score: float) -> str:
    """
    Maps the net sentiment score to a human-readable label. Uses a
    neutral buffer zone (-0.15 to +0.15) so that scores very close to
    zero aren't labeled as a strong positive or negative — consistent
    with the threshold style used in consensus_signal.py.
    """
    if net_score > 0.15:
        return "positive"
    if net_score < -0.15:
        return "negative"
    return "neutral"


# Below this article count (after filtering), the sentiment signal is
# flagged as low confidence — sample size is too small to be reliable.
# Chosen based on observed real data: AAPL/MU (large-caps) had 57-61
# relevant articles in 30 days, while CLOV/COMP (small/mid-caps) had
# only 6-8 — a full order of magnitude fewer.
MIN_ARTICLE_COUNT = 10


def news_sentiment_signal(ticker: str, company_name: str, articles: list) -> dict | None:
    """
    News Sentiment Signal — Layer 2 Quantitative Intelligence.

    Aggregates FinBERT-scored news articles (from get_news_and_sentiment)
    into a single structured sentiment signal, after filtering out
    articles that only mention the company in passing (e.g. broader
    industry/market news) rather than being genuinely about it.

    This measures MEDIA sentiment specifically — how outlets are
    currently writing about the company's own news and events. This is
    distinct from Consensus Signal (professional analyst opinion) and
    is intended to eventually sit alongside a separate Management Risk
    Sentiment Signal (how the company itself talks about its own risks
    in SEC filings) — three independent viewpoints, not to be combined
    into one score.

    Args:
        ticker: stock ticker symbol, e.g. "AAPL"
        company_name: company name for title matching, e.g. "Apple"
        articles: list of dicts from get_news_and_sentiment(), each with
                  "title", "sentiment", "score" keys at minimum

    Returns:
        dict with keys:
            - sentiment_score:  float (-1.0 to +1.0)
            - sentiment_label:  str — "positive" / "neutral" / "negative"
            - positive_count:   int
            - negative_count:   int
            - neutral_count:    int
            - total_articles:   int — count AFTER filtering
            - low_confidence:   bool — True if total_articles < MIN_ARTICLE_COUNT
            - detail:           str — plain English explanation
        or None if no company-specific articles were found after filtering
    """
    if not articles:
        return None

    relevant_articles = [
        a for a in articles
        if _is_company_specific(a.get("title", ""), ticker, company_name)
    ]

    if not relevant_articles:
        return None

    stats = _aggregate_sentiment(relevant_articles)
    label = _sentiment_label(stats["net_score"])
    low_confidence = stats["total_articles"] < MIN_ARTICLE_COUNT

    total_articles  = stats["total_articles"]
    positive_count  = stats["positive_count"]
    negative_count  = stats["negative_count"]
    neutral_count   = stats["neutral_count"]
    net_score       = stats["net_score"]

    detail = (
        f"Of {total_articles} company-specific news articles in "
        f"the past 30 days, {positive_count} were positive, "
        f"{negative_count} negative, and {neutral_count} "
        f"neutral (net sentiment score: {net_score}). Articles "
        f"were filtered to include only those naming the company in the "
        f"headline, excluding broader industry/market coverage that "
        f"merely mentions the ticker in passing."
    )
    if low_confidence:
        detail += (
            f" \u26a0\ufe0f Only {total_articles} company-specific "
            f"articles were found — small sample size, lower confidence "
            f"in this signal."
        )

    return {
        "sentiment_score": stats["net_score"],
        "sentiment_label": label,
        "positive_count":  stats["positive_count"],
        "negative_count":  stats["negative_count"],
        "neutral_count":   stats["neutral_count"],
        "total_articles":  stats["total_articles"],
        "low_confidence":  low_confidence,
        "detail":          detail,
    }
