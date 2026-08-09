from edgar import Company, set_identity
from edgar.company_reports.ten_k import TenK
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP
from src.sec_pipeline.sec_types import SecChunk

set_identity("Bo Fu bofu001@gmail.com")

SECTIONS = {
    "Item1":  ("business",              "Business Overview"),
    "Item1A": ("risk_factors",          "Risk Factors"),
    "Item7":  ("management_discussion", "MD&A"),
}

MIN_SECTION_LENGTH = 1000

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

def get_latest_tenk(ticker: str) -> tuple[TenK | None, str | None]:
    """
    Fetches the latest 10-K for this ticker from SEC EDGAR. Returns
    (None, None) on any failure (no 10-K found, network error, etc.)
    instead of raising — callers only need to check for None rather
    than writing their own try/except.
    """
    try:
        c = Company(ticker)
        # form="10-K" matches 10-K/A by prefix, and amendments carry only
        # Part III (no Item 1), so they must be excluded by exact match.
        # Iteration order is not a documented guarantee, so take the
        # newest filing_date explicitly rather than the first hit.
        filings = [f for f in c.get_filings(form="10-K") if f.form == "10-K"]
        if not filings:
            print(f"No standard 10-K found for {ticker}")
            return None, None
        else:
            latest = max(filings, key=lambda f: f.filing_date)
            return latest.obj(), str(latest.filing_date)
    except Exception as e:
        print(f"Could not check EDGAR for {ticker}: {e}")
        return None, None


def clean_text(text: str) -> str:
    # Remove page headers like "Apple Inc. | 2025 Form 10-K | 5"
    text = re.sub(r'\n\n.{1,50}\|\s*\d{4}\s*Form 10-K\s*\|\s*\d+\n\n', '\n\n', text)
    return text

def chunk_text(text: str, ticker: str, filing_type: str, filing_date: str, section_key: str, section_label: str) -> list[SecChunk]:
    docs = splitter.create_documents([text])
    return [
        SecChunk(
            ticker=ticker,
            filing_type=filing_type,
            filing_date=filing_date,
            section=section_key,
            section_label=section_label,
            text=doc.page_content,
        )
        for doc in docs
        if len(doc.page_content) >= 100
    ]


def download_and_chunk_filing(tenk: TenK, filing_date: str, ticker: str, filing_type: str = "10-K") -> list[SecChunk]:
    """
    Cleans and chunks an already-fetched TenK object. Takes tenk and
    filing_date as parameters (rather than a bare ticker) so a caller
    that already called get_latest_tenk() to check freshness can pass
    that same result straight through, instead of this function
    fetching the same filing a second time.
    """
    print(f"\n{'='*50}")
    print(f"Processing {ticker}...")

    all_chunks = []
    for section_key, (attr, label) in SECTIONS.items():
        raw = str(getattr(tenk, attr))
        if len(raw) < MIN_SECTION_LENGTH:
            print(f"  {section_key} → SKIPPED ({len(raw)} chars)")
            continue
        text = clean_text(raw)
        chunks = chunk_text(text, ticker, filing_type, filing_date, section_key, label)
        print(f"  {section_key}: {len(raw):,} chars → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"  Total: {len(all_chunks)} chunks")
    return all_chunks