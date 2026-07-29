from src.ingestion.sec_loader import ingest_sec_filing
from src.embeddings.embedder import embed_chunks
from src.vectorstore.pgvector_store import upsert_chunks, query
from src.vectorstore.types import RetrievedChunk


def retrieve(question: str, ticker: str, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Retrieve relevant chunks from pgvector for a given question and ticker.
    Used when data already exists in PostgreSQL.
    """
    # Embed the question
    embedded_question = embed_chunks([{"text": question}])
    question_vector = embedded_question[0]["embedding"]

    # Query pgvector filtered by ticker
    return query(question_vector, ticker=ticker, top_k=top_k)


def fetch_embed_store_retrieve(question: str, ticker: str, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Dynamically fetches SEC filing for a ticker not in PostgreSQL.
    Downloads, embeds, stores, then retrieves relevant chunks.
    Transaction guarantees all chunks stored or none — data integrity.
    """

    print(f"  [fetch_embed_store_retrieve] Fetching {ticker} from SEC EDGAR...")

    # Step 1: Download and chunk
    chunks = ingest_sec_filing(ticker)

    if not chunks:
        print(f"  [fetch_embed_store_retrieve] No 10-K data for {ticker} — skipping embed/store")
        return []

    print(f"  [fetch_embed_store_retrieve] Downloaded {len(chunks)} chunks")

    # Step 2: Embed
    embedded_chunks = embed_chunks(chunks)
    print(f"  [fetch_embed_store_retrieve] Embedded {len(embedded_chunks)} chunks")

    # Step 3: Store in PostgreSQL
    upsert_chunks(embedded_chunks)
    print(f"  [fetch_embed_store_retrieve] Stored in PostgreSQL")

    # Step 4: Retrieve
    return retrieve(question, ticker, top_k)