from src.sec_pipeline.sec_downloader import download_and_chunk_filing
from edgar.company_reports.ten_k import TenK
from src.sec_pipeline.embedder import embed_chunks
from src.sec_pipeline.pgvector_store import insert_chunks, delete_chunks, query
from src.sec_pipeline.sec_types import RetrievedChunk


def retrieve(question: str, ticker: str) -> list[RetrievedChunk]:
    """
    Retrieve relevant chunks from pgvector for a given question and ticker.
    Used when data already exists in PostgreSQL.
    """
    # Embed the question
    embedded_question = embed_chunks([{"text": question}])
    question_vector = embedded_question[0]["embedding"]

    # Query pgvector filtered by ticker
    return query(question_vector, ticker=ticker)


def fetch_embed_store_retrieve(question: str, ticker: str, tenk: TenK, filing_date: str) -> list[RetrievedChunk]:
    """
    Embeds, stores, then retrieves relevant chunks for an already-
    fetched TenK object. tenk and filing_date come from the caller's
    own get_latest_tenk() call — this function does not fetch the
    filing itself, so callers must fetch it first (e.g. to check
    freshness) and pass the result straight through here, rather
    than fetching the same filing twice.
    Transaction guarantees all chunks stored or none — data integrity.
    """

    print(f"  [fetch_embed_store_retrieve] Processing {ticker}...")

    # Step 1: Chunk the already-fetched filing
    chunks = download_and_chunk_filing(tenk, filing_date, ticker)

    if not chunks:
        print(f"  [fetch_embed_store_retrieve] No 10-K data for {ticker} — skipping embed/store")
        return []

    print(f"  [fetch_embed_store_retrieve] Downloaded {len(chunks)} chunks")

    # Step 2: Embed
    embedded_chunks = embed_chunks(chunks)
    print(f"  [fetch_embed_store_retrieve] Embedded {len(embedded_chunks)} chunks")

    # Step 3: Delete any existing chunks for this ticker, then store
    # the freshly downloaded ones — unconditional, so old and new
    # chunks never coexist (a no-op if nothing was stored before).
    delete_chunks(ticker)
    insert_chunks(embedded_chunks)
    print(f"  [fetch_embed_store_retrieve] Stored in PostgreSQL")

    # Step 4: Retrieve
    return retrieve(question, ticker)