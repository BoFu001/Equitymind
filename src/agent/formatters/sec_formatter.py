"""
src/agent/formatters/sec_formatter.py

Formats SEC filing chunks (retrieved via src/readers/sec_retrieval.py's
pgvector similarity search) for the LLM.

Split out from the former single-file formatters.py on 2026-07-27,
alongside the per-signal formatters. Not tied to any quant signal —
this is raw SEC 10-K text, retrieved independently of Valuation/Risk/
Quality/Consensus/Momentum/News Sentiment.
"""


def format_sec_chunks(chunks: list, ticker: str) -> str:
    sec_context = f"\n{ticker} SEC Filing:\n"
    for i, chunk in enumerate(chunks):
        source = f"{ticker}_{chunk['chunk']['filing_type']}_{chunk['chunk']['section']}_{chunk['chunk']['filing_date']}"
        sec_context += f"[Source {i+1}: {source} | Score: {chunk['score']:.2f}]\n"
        sec_context += chunk["chunk"]["text"] + "\n"
    return sec_context
