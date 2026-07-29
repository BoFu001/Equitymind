"""
src/agent/formatters/risk_formatter.py

Formats risk_signal() results for the LLM — see
src/quant/risk_signal.py for the data this consumes.

Split out from the former single-file formatters.py on 2026-07-27,
alongside consensus_formatter.py, valuation_formatter.py,
snapshot_formatter.py, momentum_formatter.py,
news_sentiment_formatter.py, and quality_formatter.py.

Unlike quality_formatter.py (where each sub-signal's detail string was
already self-contained), risk_signal.py's per-sub-signal fields
(raw_beta, annualised_return, volatility_annual, peak_date,
trough_date, peak_price, trough_price, etc.) were previously only
summarized as a single score per line, with the full detail buried in
the bottom-level "detail" string rather than exposed as independent
fields — same gap as consensus_formatter.py / valuation_formatter.py
had before 2026-07-27. Fixed here: each of the 4 sub-signals
(Beta/Sharpe/VaR/Max Drawdown) now shows its own raw data line.
"""


def format_risk(risk: dict | None) -> str:
    """
    Four independent sub-signals, NOT combined into a single score
    (Beta/Sharpe/VaR/Max Drawdown each answer a different risk question
    — averaging them would hide which dimension matters, e.g. a strong
    Sharpe Ratio can mask a catastrophic Max Drawdown).
    """
    if not risk:
        return "  Risk: Insufficient data.\n"

    text = "  Risk:\n"

    if risk.get("beta"):
        b = risk["beta"]
        text += (
            f"    Beta: raw={b['raw_beta']}, Blume-adjusted={b['adjusted_beta']} "
            f"(score={b['beta_score']})\n"
        )
    else:
        text += "    Beta: Unavailable (market benchmark data missing).\n"

    s = risk["sharpe"]
    text += (
        f"    Sharpe Ratio: {s['sharpe_ratio']} (score={s['sharpe_score']}) — "
        f"annualised return={s['annualised_return']*100:.2f}%, "
        f"volatility={s['volatility_annual']*100:.2f}%\n"
    )

    v = risk["var"]
    text += f"    VaR (95%, 1-day): {v['var_95']*100:.2f}% (score={v['var_score']})\n"

    md = risk["max_drawdown"]
    text += (
        f"    Max Drawdown: {md['max_drawdown']*100:.2f}% (score={md['drawdown_score']}) — "
        f"peak ${md['peak_price']} on {md['peak_date']}, "
        f"trough ${md['trough_price']} on {md['trough_date']}\n"
    )

    text += f"    {risk['detail']}\n"

    if risk.get("low_confidence"):
        text += "    Note: Risk signal based on less than 1 year of price history — lower confidence.\n"
    if risk.get("beta") is None:
        text += "    Note: Beta could not be computed — market benchmark data unavailable.\n"
    if risk.get("max_drawdown", {}).get("stress_tested") is False:
        text += "    Note: This stock has not experienced a significant decline in the observed window — risk may be understated.\n"

    return text
