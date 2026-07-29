"""
src/agent/formatters/__init__.py

Shared formatters for converting structured data into LLM-readable
strings. Used by fetch_all_data.py and generate_report.py to ensure
consistent data presentation.

Was a single file (formatters.py) until 2026-07-27, when each
formatter was split into its own file, one per data source/signal —
same principle already applied to src/tools/ (snapshot_reader.py,
risk_reader.py, consensus_reader.py, etc.). This file now contains no
logic of its own, only re-exports, so every existing
`from src.agent.formatters import format_xxx` keeps working unchanged;
callers don't need to know or care which submodule a given formatter
lives in.
"""

from src.agent.formatters.consensus_formatter import format_consensus
from src.agent.formatters.valuation_formatter import format_valuation
from src.agent.formatters.snapshot_formatter import format_stock_snapshot
from src.agent.formatters.momentum_formatter import format_momentum
from src.agent.formatters.news_sentiment_formatter import format_news_sentiment
from src.agent.formatters.quality_formatter import format_quality
from src.agent.formatters.risk_formatter import format_risk
from src.agent.formatters.sec_formatter import format_sec_chunks
from src.agent.formatters.conversation_formatter import format_conversation_context
from src.agent.formatters.financial_history_formatter import format_financial_history
from src.agent.formatters.quant_signals_formatter import format_quant_signals
