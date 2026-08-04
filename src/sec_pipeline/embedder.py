import time
from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_BATCH_SIZE
from src.sec_pipeline.sec_types import SecChunk, EmbeddedSecChunk

client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_chunks(chunks: list[SecChunk]) -> list[EmbeddedSecChunk]:
    """
    Takes chunks from sec_downloader.
    Adds 'embedding' field to each chunk.
    Returns chunks with embeddings.
    """
    print(f"Embedding {len(chunks)} chunks...")

    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        for j, item in enumerate(response.data):
            batch[j]["embedding"] = item.embedding

        print(f"  {min(i + EMBEDDING_BATCH_SIZE, len(chunks))}/{len(chunks)} done")
        time.sleep(0.1)  # be polite to the API

    print(f"Embedding complete. Total: {len(chunks)} chunks")
    return chunks