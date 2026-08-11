"""
scripts/update_stock_overviews.py

Generates a plain-English business overview for every ticker in the
stock_universe table from its latest 10-K Item 1, embeds it, and writes
both to overview_text / overview_embedding. Discovery matches a user's
open-ended theme ("robotics", "banking", "meat industry") against these
embeddings — an approach chosen because sector labels cannot be
enumerated in advance, and single-label classification misassigns
multi-business companies (the same GICS problem that forced peer-group
benchmarks in valuation_signal.py).

Every ticker is checked on every run, but an overview is only
regenerated when the company has filed a newer 10-K than the one it was
built from — overview_filing_date records which filing that was. The
comparison happens before the filing is parsed, so an unchanged ticker
costs a single EDGAR metadata request and no LLM call at all. Tickers
that failed last time are retried automatically (SPCX, a June 2026 IPO,
will succeed once its first 10-K is filed).

Four content checks run BEFORE the LLM is called, because a minimum
length alone is not sufficient — measured across all 250 tickers on
2026-08-09:
    - empty section (4 tickers: DVN, FANG, KMI, PSX) — energy filers
      title the section "Items 1 and 2. Business and Properties", which
      the parser does not match, returning 0 chars. Caught by length.
    - cross-reference index (1 ticker: GE) — Item 1 is a page-number
      table pointing into other sections, ~5.7k chars of "incorporated
      by reference". Passes a length check but contains no business
      description, so the LLM would write a fluent, empty overview.
    - unparsed HTML (1 ticker: C) — 1.78M chars of raw div markup,
      roughly 450k tokens, several dollars in a single call.
    - oversized section (0 tickers) — a backstop for any other runaway
      extraction; the longest legitimate Item 1 measured was 199k.
Both GE and C would otherwise succeed, and a wrong overview is worse
than a missing one: it enters the vector index and pollutes retrieval.
Tickers failing any check keep a NULL overview and are skipped by
Discovery's theme search (their quant_signals screening is unaffected).

Usage:
    python scripts/update_stock_overviews.py
"""

import sys
import time
from pathlib import Path

import psycopg2
from edgar import Company, set_identity
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sec_pipeline.embedder import EMBEDDING_MODEL
from config import DATABASE_URL, OPENAI_API_KEY, LLM_MODEL

set_identity("Bo Fu bofu001@gmail.com")
client = OpenAI(api_key=OPENAI_API_KEY)

# LLM_MODEL, not LLM_MODEL_LIGHT: the lighter model tagged Amazon as
# retail only in earlier labelling tests, missing AWS entirely.
# EMBEDDING_MODEL is imported rather than redeclared so overview vectors
# and Discovery's query vectors always share the same space.

SEC_THROTTLE_SECONDS = 0.5

MIN_BUSINESS_CHARS = 1000
MAX_BUSINESS_CHARS = 250_000  # longest legitimate Item 1 measured is ABNB at 199k

OVERVIEW_PROMPT = """Based on the following official 10-K business description, \
write a concise company overview (150-200 words) that captures:
- Core business lines (what they actually do)
- Market position (leader, challenger, niche player)
- Public perception (what ordinary people/media commonly associate this company with)
- Industry/theme classification (multiple applicable tags, e.g. technology, \
e-commerce, cloud computing)

Two rules govern every claim in the overview:

1. RETAIN what the text states. If the 10-K text names a specific product, \
brand, technology, strategic initiative, or figure that is relevant to the \
company's business or identity, include it — even if it appears only once, \
appears in a long list alongside other items, or appears late in a long \
filing. Do not omit a real, named detail just to keep the overview short; \
prioritize which details to keep, but do not silently drop ones that are \
clearly material to the company's business (flagship products, named \
strategic principles, core brands).

2. DO NOT ADD what the text does not state. Never supply a specific number, \
percentage, ranking, or comparison that is not written in the text below, \
even if you believe you know the correct or current figure from other \
sources. If the text doesn't give a number, describe the fact qualitatively \
instead of inventing or recalling one.

10-K Business Description:
{business_text}

Write the overview now, in plain, natural language (not bullet points):"""


def _load_target_tickers(cursor) -> list[tuple[str, str | None]]:
    """Returns (ticker, overview_filing_date) for the whole universe —
    the stored filing_date is what decides whether an overview needs
    regenerating, so it is fetched here rather than queried per ticker."""
    cursor.execute(
        "SELECT ticker, overview_filing_date FROM stock_universe ORDER BY ticker"
    )
    return [(row[0], row[1]) for row in cursor.fetchall()]


def _is_cross_reference_index(text: str) -> bool:
    """GE-style filings replace Item 1 with a page-number index pointing
    into other sections — long enough to pass a length check, with no
    business description in it."""
    lowered = text.lower()
    return (
        "cross reference index" in lowered[:2000]
        or lowered.count("incorporated by reference") > 5
    )


def _is_raw_html(text: str) -> bool:
    """Some filings (C) return unparsed HTML instead of extracted text.
    A real Item 1 does not open with markup."""
    head = text[:500]
    return head.count("<div") > 3 or head.count("style=") > 3


def fetch_business_text(ticker: str, stored_filing_date: str | None) -> tuple[str | None, str | None]:
    """Returns (business_text, filing_date) for the latest 10-K, or
    (None, None) if the stored overview already came from that filing,
    or if the section is unusable. Prints the reason so a full run
    leaves a readable record of what was skipped and why.

    The date comparison happens before .obj(), the only expensive call
    here — an unchanged 10-K costs one metadata request and nothing
    else. Same check-before-you-fetch pattern as
    update_financial_history.py's _has_new_data()."""
    try:
        # form="10-K" matches 10-K/A by prefix, and amendments carry only
        # Part III (no Item 1). Exact-match, then take the newest
        # filing_date — iteration order is not a documented guarantee.
        filings = [f for f in Company(ticker).get_filings(form="10-K") if f.form == "10-K"]
    except Exception as e:
        print(f"    Skipping {ticker}: EDGAR lookup failed — {e}")
        return None, None

    if not filings:
        print(f"    Skipping {ticker}: no 10-K on file")
        return None, None
    else:
        latest = max(filings, key=lambda f: f.filing_date)

    filing_date = str(latest.filing_date)
    if stored_filing_date == filing_date:
        print(f"    {ticker}: overview already built from the {filing_date} 10-K")
        return None, None

    try:
        text = str(getattr(latest.obj(), "business", "") or "")
    except Exception as e:
        print(f"    Skipping {ticker}: could not parse filing — {e}")
        return None, None

    if len(text) < MIN_BUSINESS_CHARS:
        print(f"    Skipping {ticker}: Item 1 only {len(text)} chars")
        return None, None
    elif _is_raw_html(text):
        print(f"    Skipping {ticker}: Item 1 is unparsed HTML ({len(text):,d} chars)")
        return None, None
    elif _is_cross_reference_index(text):
        print(f"    Skipping {ticker}: Item 1 is a cross-reference index")
        return None, None
    elif len(text) > MAX_BUSINESS_CHARS:
        print(f"    Skipping {ticker}: Item 1 unexpectedly large ({len(text):,d} chars)")
        return None, None
    else:
        return text, filing_date


def generate_overview(business_text: str) -> str:
    """Summarises Item 1 into a 150-200 word overview. The full section
    is passed, not a prefix — truncating risks cutting a business line
    that appears late in the text."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": OVERVIEW_PROMPT.format(business_text=business_text)}
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def embed_overview(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def update_stock_overviews():
    print("EquityMind — Stock Overview Updater")
    print("=" * 50)

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    targets = _load_target_tickers(cursor)
    print(f"\nUniverse size: {len(targets)} tickers")
    print("Checking each ticker's latest 10-K against its stored overview...\n")

    written = 0
    skipped = []

    for i, (ticker, stored_filing_date) in enumerate(targets):
        print(f"  [{i+1}/{len(targets)}] {ticker}...", flush=True)

        business_text, filing_date = fetch_business_text(ticker, stored_filing_date)
        time.sleep(SEC_THROTTLE_SECONDS)

        if business_text is None:
            # An up-to-date overview also returns None — only count it as
            # skipped when there is no overview to fall back on.
            if stored_filing_date is None:
                skipped.append(ticker)
            continue

        try:
            overview = generate_overview(business_text)
            embedding = embed_overview(overview)
        except Exception as e:
            print(f"    Skipping {ticker}: generation failed — {e}")
            skipped.append(ticker)
            continue

        try:
            cursor.execute(
                """
                UPDATE stock_universe
                SET overview_text = %s,
                    overview_embedding = %s,
                    overview_filing_date = %s
                WHERE ticker = %s
                """,
                # str(), not the raw list: psycopg2 renders a Python list
                # as a Postgres array literal, which vector(1536) rejects.
                # pgvector parses the '[0.1, 0.2, ...]' text form directly.
                (overview, str(embedding), filing_date, ticker),
            )
            conn.commit()  # per ticker — a 30-minute run should never lose work
            written += 1
        except Exception as e:
            print(f"    DB write failed for {ticker}: {e}")
            skipped.append(ticker)
            conn.rollback()

    cursor.close()
    conn.close()

    print(f"\n✓ stock_universe table updated — {written} overview(s) written")
    if skipped:
        print(f"  No usable Item 1 for {len(skipped)}: {', '.join(skipped)}")
        print("  These keep a NULL overview and are excluded from theme search.")


if __name__ == "__main__":
    update_stock_overviews()
