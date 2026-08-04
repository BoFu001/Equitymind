"""
src/vectorstore/types.py

Type definitions for SEC filing data flow.

Data flow:
    ingest_sec_filing()  → list[SecChunk]
    embed_chunks()       → list[EmbeddedSecChunk]
    upsert_chunks()      → None (stores to PostgreSQL)
    query()              → list[RetrievedChunk]
"""

from typing import TypedDict


class SecChunk(TypedDict):
    """
    A single SEC filing chunk after download and splitting.
    Produced by ingest_sec_filing(), consumed by embed_chunks().
    """
    ticker:        str
    filing_type:   str
    filing_date:   str
    section:       str
    section_label: str
    text:          str


class EmbeddedSecChunk(SecChunk, total=False):
    """
    SecChunk with embedding added.
    Produced by embed_chunks(), consumed by upsert_chunks().
    """
    embedding: list[float]


class RetrievedChunk(TypedDict):
    """
    A chunk retrieved from pgvector with similarity score.
    Produced by query(), consumed by nodes.py report generators.
    Score represents cosine similarity between chunk and question.
    """
    chunk: SecChunk
    score: float