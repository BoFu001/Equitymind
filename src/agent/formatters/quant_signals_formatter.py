"""
src/agent/formatters/quant_signals_formatter.py

Concatenates the 6 quant signal formatters into a single block per
ticker — the only formatter left in __init__.py until now.

Split out on 2026-07-27, alongside the other per-signal formatters,
so __init__.py contains no logic of its own (just re-exports),
matching the convention used elsewhere in the project (e.g.
src/tools/__init__.py).

Deliberately does NOT include format_stock_snapshot, format_sec_chunks,
format_conversation_context, or format_financial_history — those are
independent data sources, not quant signals, and are called separately
by generate_report.py.

NOTE: this "always concatenate all 6" design is a product of the
current "fetch everything unconditionally" architecture (see
fetch_data.py's docstring) — it is expected to change once a
future "which data does this question need" routing node
(determine_data_needs) exists. Once that node can decide, e.g., that
a question needs only Consensus, this function's assumption that all
6 signals always belong together in one block will no longer hold —
generate_report.py will likely call the needed format_xxx() functions
directly instead of going through a fixed bundler like this one. Not
refactored now because the routing node itself doesn't exist yet;
noted here so this isn't mistaken for a permanent design.

Grouped into two categories (2026-07-27), reflecting a real
methodological distinction, not just a display preference:
  - Objective Financial Metrics (Valuation, Momentum, Risk, Quality):
    computed purely from financial figures and price history — no
    human opinion involved.
  - Market Sentiment & Opinion (News Sentiment, Consensus): reflects
    human judgment — professional analyst views (Consensus) or media
    tone (News Sentiment) — not an objective calculation. Both already
    carry their own bias disclosures in their detail text
    (consensus_signal.py's "systematic optimism bias" note; FinBERT's
    tone scoring is itself an interpretation, not a fact). Separating
    them from the objective metrics, rather than listing all 6 signals
    as an undifferentiated block, makes this distinction visible to
    the LLM (and, downstream, the user) rather than implicit.
"""

from src.agent.formatters.valuation_formatter import format_valuation
from src.agent.formatters.momentum_formatter import format_momentum
from src.agent.formatters.risk_formatter import format_risk
from src.agent.formatters.quality_formatter import format_quality
from src.agent.formatters.news_sentiment_formatter import format_news_sentiment
from src.agent.formatters.consensus_formatter import format_consensus


def format_quant_signals(signals: dict, ticker: str) -> str:
    """
    Formats all six quant signals for one ticker into a single block,
    grouped by objective/subjective (see module docstring).
    """
    text = f"\n{ticker} Quantitative Signals:\n"

    text += "\n  Objective Financial Metrics:\n"
    text += format_valuation(signals.get("valuation"))
    text += format_momentum(signals.get("momentum"))
    text += format_risk(signals.get("risk"))
    text += format_quality(signals.get("quality"))

    text += "\n  Market Sentiment & Opinion (subjective, not objective calculations):\n"
    text += format_news_sentiment(signals.get("news_sentiment"))
    text += format_consensus(signals.get("consensus"))

    return text
