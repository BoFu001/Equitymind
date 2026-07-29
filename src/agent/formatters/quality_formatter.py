"""
src/agent/formatters/quality_formatter.py

Formats quality_signal() results for the LLM — see
src/quant/quality_signal.py for the data this consumes.

Split out from the former single-file formatters.py on 2026-07-27,
alongside consensus_formatter.py, valuation_formatter.py,
snapshot_formatter.py, momentum_formatter.py, and
news_sentiment_formatter.py. No content changes — reviewed 2026-07-27
and found already detailed: each of the 9 F-Score sub-signals'
detail string already includes the underlying raw figures (e.g.
"ROA 31.2% vs prior year 25.7%"), not just a pass/fail judgment, so
there was no gap to fill (unlike consensus_formatter.py /
valuation_formatter.py, which needed raw inputs added).
"""


def format_quality(quality: dict | None) -> str:
    """
    Shows the F-Score summary plus all 9 signals grouped by outcome
    (met / not met / insufficient data), so the LLM and the user can see
    exactly WHICH signals drove the score, not just the aggregate count
    — needed to judge whether a low score reflects genuine deterioration
    or a deliberate short-term trade-off (per the detail note below).
    """
    if not quality:
        return "  Quality: Insufficient data.\n"
    text = f"  Quality: {quality['quality_label']} (F-Score={quality['f_score_raw']}/{quality['signals_evaluated']}, score={quality['quality_score']})\n"
    text += f"  {quality['detail']}\n"
    if quality.get("signals_evaluated", 9) < 9:
        text += f"  Note: Only {quality['signals_evaluated']} of 9 F-Score signals could be evaluated due to missing financial data.\n"

    breakdown = quality.get("breakdown") or {}
    passed  = [(k, v) for k, v in breakdown.items() if v["score"] == 1]
    failed  = [(k, v) for k, v in breakdown.items() if v["score"] == 0]
    missing = [(k, v) for k, v in breakdown.items() if v["score"] is None]

    if passed:
        text += f"  Signals met ({len(passed)}):\n"
        for name, v in passed:
            text += f"    - {name}: {v['detail']}\n"
    if failed:
        text += f"  Signals not met ({len(failed)}):\n"
        for name, v in failed:
            text += f"    - {name}: {v['detail']}\n"
    if missing:
        text += f"  Signals with insufficient data ({len(missing)}):\n"
        for name, v in missing:
            text += f"    - {name}: {v['detail']}\n"
    return text
