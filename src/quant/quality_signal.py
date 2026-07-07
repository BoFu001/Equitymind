"""
src/quant/quality_signal.py

Quality Signal Engine — Layer 2 Quantitative Intelligence.

Implements Piotroski's F-Score (Piotroski, 2000) — 9 binary signals
assessing a company's fundamental financial health, independent of its
stock price. Unlike Valuation, Momentum, and Risk (which all depend on
price), Quality is purely about the underlying business: is it
profitable, generating real cash, reducing leverage, and improving
operational efficiency — regardless of whether the market currently
prices it cheap or expensive.

Score range: -1.0 (very low quality) to +1.0 (very high quality)
Label:       "high" / "medium" / "low"

This is a pure function — it takes already-fetched data (from
market_data.get_quality_inputs()) as input and performs no I/O itself,
same pattern as valuation_signal.py, momentum_signal.py, and
risk_signal.py.

Known limitation (disclosed, not corrected): F-Score is a backward-looking
snapshot of year-over-year accounting changes. It cannot distinguish a
company whose fundamentals are genuinely deteriorating from one that is
deliberately sacrificing short-term margins for a long-term strategic
position (e.g. aggressive market-share expansion). A low F-Score should
be read alongside the company's own narrative (e.g. SEC filings), not
treated as a standalone verdict.

Academic reference:
    Piotroski, J. (2000), 'Value Investing: The Use of Historical
    Financial Statement Information to Separate Winners from Losers',
    Journal of Accounting Research.
"""

from src.quant.quality_signal_config import (
    MAX_F_SCORE,
    HIGH_QUALITY_THRESHOLD,
    LOW_QUALITY_THRESHOLD,
)


# ─────────────────────────────────────────────
# 9 individual Piotroski signals
# Each returns 1 (criterion met), 0 (not met), or None (cannot be
# evaluated — a required input is missing).
# ─────────────────────────────────────────────

def _signal_positive_net_income(current: dict) -> tuple[int | None, str]:
    """Signal 1: Net income is positive."""
    ni = current.get("net_income")
    if ni is None:
        return None, "Net income data unavailable."
    score = 1 if ni > 0 else 0
    return score, f"Net income {'positive' if score else 'negative/zero'} (${ni/1e9:.2f}B)"


def _signal_positive_operating_cash_flow(current: dict) -> tuple[int | None, str]:
    """Signal 2: Operating cash flow is positive."""
    ocf = current.get("operating_cash_flow")
    if ocf is None:
        return None, "Operating cash flow data unavailable."
    score = 1 if ocf > 0 else 0
    return score, f"Operating cash flow {'positive' if score else 'negative/zero'} (${ocf/1e9:.2f}B)"


def _signal_roa_improving(current: dict, prior: dict) -> tuple[int | None, str]:
    """Signal 3: Return on Assets (ROA) improved year-over-year."""
    ni_cur, ta_cur = current.get("net_income"), current.get("total_assets")
    ni_pri, ta_pri = prior.get("net_income"), prior.get("total_assets")
    if None in (ni_cur, ta_cur, ni_pri, ta_pri) or ta_cur == 0 or ta_pri == 0:
        return None, "ROA comparison unavailable (missing net income or total assets)."
    roa_cur = ni_cur / ta_cur
    roa_pri = ni_pri / ta_pri
    score = 1 if roa_cur > roa_pri else 0
    return score, f"ROA {roa_cur*100:.1f}% vs prior year {roa_pri*100:.1f}%"


def _signal_cash_flow_exceeds_net_income(current: dict) -> tuple[int | None, str]:
    """Signal 4: Operating cash flow exceeds net income (earnings quality check)."""
    ocf, ni = current.get("operating_cash_flow"), current.get("net_income")
    if ocf is None or ni is None:
        return None, "Cash flow vs net income comparison unavailable."
    score = 1 if ocf > ni else 0
    return score, f"Operating cash flow (${ocf/1e9:.2f}B) {'exceeds' if score else 'does not exceed'} net income (${ni/1e9:.2f}B)"


def _signal_leverage_decreasing(current: dict, prior: dict) -> tuple[int | None, str]:
    """Signal 5: Long-term debt ratio decreased year-over-year."""
    ltd_cur, ta_cur = current.get("long_term_debt"), current.get("total_assets")
    ltd_pri, ta_pri = prior.get("long_term_debt"), prior.get("total_assets")
    if None in (ltd_cur, ta_cur, ltd_pri, ta_pri) or ta_cur == 0 or ta_pri == 0:
        return None, "Leverage comparison unavailable (missing debt or total assets)."
    ratio_cur = ltd_cur / ta_cur
    ratio_pri = ltd_pri / ta_pri
    score = 1 if ratio_cur < ratio_pri else 0
    return score, f"Long-term debt ratio {ratio_cur*100:.1f}% vs prior year {ratio_pri*100:.1f}%"


def _signal_current_ratio_improving(current: dict, prior: dict) -> tuple[int | None, str]:
    """Signal 6: Current ratio (short-term liquidity) improved year-over-year."""
    ca_cur, cl_cur = current.get("current_assets"), current.get("current_liabilities")
    ca_pri, cl_pri = prior.get("current_assets"), prior.get("current_liabilities")
    if None in (ca_cur, cl_cur, ca_pri, cl_pri) or cl_cur == 0 or cl_pri == 0:
        return None, "Current ratio comparison unavailable."
    ratio_cur = ca_cur / cl_cur
    ratio_pri = ca_pri / cl_pri
    score = 1 if ratio_cur > ratio_pri else 0
    return score, f"Current ratio {ratio_cur:.2f} vs prior year {ratio_pri:.2f}"


def _signal_no_share_dilution(current: dict, prior: dict) -> tuple[int | None, str]:
    """
    Signal 7: No new shares issued (shares outstanding did not increase).
    A company that reduced share count via buybacks also satisfies this —
    fewer shares is at least as good for existing shareholders as unchanged.

    Follows Piotroski's original binary definition strictly: any increase,
    however small, fails this signal. This is a known limitation of the
    original methodology — it does not distinguish a large dilutive
    issuance from a negligible one (e.g. a fractional-percent change), so
    a company with an economically trivial increase can still fail this
    signal. When the change is very small, the detail notes the magnitude
    so the number isn't misread as a large dilution event, without
    speculating on its cause.
    """
    shares_cur = current.get("shares_outstanding")
    shares_pri = prior.get("shares_outstanding")
    if shares_cur is None or shares_pri is None:
        return None, "Share count comparison unavailable."
    score = 1 if shares_cur <= shares_pri else 0
    change = "decreased" if shares_cur < shares_pri else ("unchanged" if shares_cur == shares_pri else "increased")
    pct_change = ((shares_cur - shares_pri) / shares_pri) * 100
    magnitude_note = f" ({abs(pct_change):.5f}% change — a very small magnitude)" if abs(pct_change) < 0.5 and score == 0 else ""
    return score, f"Shares outstanding {change} ({shares_cur/1e9:.2f}B vs prior {shares_pri/1e9:.2f}B){magnitude_note}"


def _signal_gross_margin_improving(current: dict, prior: dict) -> tuple[int | None, str]:
    """Signal 8: Gross margin improved year-over-year."""
    gp_cur, rev_cur = current.get("gross_profit"), current.get("total_revenue")
    gp_pri, rev_pri = prior.get("gross_profit"), prior.get("total_revenue")
    if None in (gp_cur, rev_cur, gp_pri, rev_pri) or rev_cur == 0 or rev_pri == 0:
        return None, "Gross margin comparison unavailable."
    margin_cur = gp_cur / rev_cur
    margin_pri = gp_pri / rev_pri
    score = 1 if margin_cur > margin_pri else 0
    return score, f"Gross margin {margin_cur*100:.1f}% vs prior year {margin_pri*100:.1f}%"


def _signal_asset_turnover_improving(current: dict, prior: dict) -> tuple[int | None, str]:
    """Signal 9: Asset turnover ratio improved year-over-year."""
    rev_cur, ta_cur = current.get("total_revenue"), current.get("total_assets")
    rev_pri, ta_pri = prior.get("total_revenue"), prior.get("total_assets")
    if None in (rev_cur, ta_cur, rev_pri, ta_pri) or ta_cur == 0 or ta_pri == 0:
        return None, "Asset turnover comparison unavailable."
    turnover_cur = rev_cur / ta_cur
    turnover_pri = rev_pri / ta_pri
    score = 1 if turnover_cur > turnover_pri else 0
    return score, f"Asset turnover {turnover_cur:.2f} vs prior year {turnover_pri:.2f}"


# ─────────────────────────────────────────────
# MAIN FUNCTION: quality_signal
# ─────────────────────────────────────────────

def quality_signal(quality_inputs: dict | None) -> dict | None:
    """
    Compute a quality signal (Piotroski F-Score) from financial statement data.

    Pure function — takes already-fetched data, performs no I/O. Data
    fetching is done separately by market_data.get_quality_inputs(), called
    by quant_engine.py before this function runs (same pattern as
    valuation_signal.py, momentum_signal.py, and risk_signal.py).

    Degradation strategy: if fewer than 2 fiscal years of statements are
    available at all, get_quality_inputs() itself returns None and this
    function does too (none of the 9 signals can be computed). Within a
    valid 2-year window, individual signals are allowed to be missing
    (e.g. a line item wasn't reported) — the F-Score is then computed as
    a fraction of the signals that COULD be evaluated, scaled up to the
    standard 0-9 range, rather than assuming a missing signal failed.

    Args:
        quality_inputs: dict from get_quality_inputs(), with "current" and
                         "prior" keys, each a dict of raw financial figures.

    Returns:
        dict with keys:
            - f_score:          float (0-9), scaled if some signals were
                                 unavailable
            - f_score_raw:      int, count of signals actually satisfied
                                 (out of however many were evaluable)
            - signals_evaluated: int, how many of the 9 signals had enough
                                 data to be judged
            - quality_score:    float (-1.0 to +1.0)
            - quality_label:    str — "high" / "medium" / "low"
            - breakdown:        dict — each of the 9 signal names mapped to
                                 (score, detail) for transparency
            - detail:           str — plain English explanation
        or None if quality_inputs is None (fewer than 2 fiscal years available).
    """
    if quality_inputs is None:
        return None

    current = quality_inputs.get("current") or {}
    prior   = quality_inputs.get("prior") or {}

    signal_functions = {
        "positive_net_income":          lambda: _signal_positive_net_income(current),
        "positive_operating_cash_flow": lambda: _signal_positive_operating_cash_flow(current),
        "roa_improving":                lambda: _signal_roa_improving(current, prior),
        "cash_flow_exceeds_net_income": lambda: _signal_cash_flow_exceeds_net_income(current),
        "leverage_decreasing":          lambda: _signal_leverage_decreasing(current, prior),
        "current_ratio_improving":      lambda: _signal_current_ratio_improving(current, prior),
        "no_share_dilution":            lambda: _signal_no_share_dilution(current, prior),
        "gross_margin_improving":       lambda: _signal_gross_margin_improving(current, prior),
        "asset_turnover_improving":     lambda: _signal_asset_turnover_improving(current, prior),
    }

    breakdown = {}
    for name, fn in signal_functions.items():
        score, detail = fn()
        breakdown[name] = {"score": score, "detail": detail}

    evaluable_scores = [v["score"] for v in breakdown.values() if v["score"] is not None]
    signals_evaluated = len(evaluable_scores)

    if signals_evaluated == 0:
        return None  # no financial data could be evaluated at all

    f_score_raw = sum(evaluable_scores)

    # Scale to the standard 0-9 range if fewer than 9 signals were evaluable,
    # so a company with (e.g.) 7/7 evaluable signals satisfied doesn't look
    # artificially worse than one with 9/9 — both represent "all available
    # evidence is positive".
    f_score = (f_score_raw / signals_evaluated) * MAX_F_SCORE

    # Normalise to [-1, +1]: F-Score=0 -> -1, F-Score=9 -> +1, F-Score=4.5 -> 0
    quality_score = round((f_score / MAX_F_SCORE) * 2 - 1, 4)

    # ── Label ──────────────────────────────────────────────────────────
    if f_score >= HIGH_QUALITY_THRESHOLD:
        quality_label = "high"
    elif f_score <= LOW_QUALITY_THRESHOLD:
        quality_label = "low"
    else:
        quality_label = "medium"

    # ── Detail ─────────────────────────────────────────────────────────
    detail_parts = [
        f"F-Score: {f_score_raw}/{signals_evaluated} signals met "
        f"(scaled to {round(f_score, 1)}/9)."
    ]
    if signals_evaluated < 9:
        detail_parts.append(
            f"Note: only {signals_evaluated} of 9 signals could be evaluated "
            f"due to missing financial data."
        )
    detail_parts.append(
        "This score reflects year-over-year accounting changes only — it "
        "cannot distinguish genuine fundamental deterioration from a "
        "deliberate short-term trade-off (e.g. sacrificing margin to gain "
        "market share). Consider alongside the company's own disclosures."
    )

    return {
        "f_score":           round(f_score, 2),
        "f_score_raw":       f_score_raw,
        "signals_evaluated": signals_evaluated,
        "quality_score":     quality_score,
        "quality_label":     quality_label,
        "breakdown":         breakdown,
        "detail":            " ".join(detail_parts),
    }
