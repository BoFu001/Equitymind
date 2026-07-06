"""
src/quant/risk_signal.py

Risk Signal Engine — Layer 2 Quantitative Intelligence.

Assesses how risky a stock is using four industry-standard metrics:
    1. Beta             — sensitivity to overall market movements (weight: 30%)
    2. Sharpe Ratio     — risk-adjusted return efficiency (weight: 30%)
    3. VaR (95%)        — potential single-day loss in normal conditions (weight: 20%)
    4. Max Drawdown     — worst historical peak-to-trough decline (weight: 20%)

Score range: -1.0 (very high risk) to +1.0 (very low risk)
Label:       "low" / "medium" / "high"

This is a pure function — it takes already-fetched data as input and
performs no I/O itself. Data fetching lives in market_data.get_risk_inputs(),
called separately by quant_engine.py, keeping this file testable without
mocking any network calls (same pattern as valuation_signal.py and
momentum_signal.py).

Academic references:
    - Beta / Blume Adjustment: Blume, M. (1971), 'On the Assessment of Risk'
    - Sharpe Ratio: Sharpe, W. (1966), 'Mutual Fund Performance'
    - VaR (historical simulation): standard retail risk-reporting convention
    - Max Drawdown: standard risk metric, no distributional assumptions
"""

import numpy as np
import pandas as pd

from src.quant.risk_signal_config import (
    MIN_TRADING_DAYS,
    LOW_CONFIDENCE_THRESHOLD,
    BLUME_RAW_WEIGHT,
    BLUME_MARKET_WEIGHT,
    BETA_EXTREME_ANCHOR,
    SHARPE_TANH_DIVISOR,
    RISK_FREE_RATE_FALLBACK,
    VAR_EXTREME_ANCHOR,
    VAR_CONFIDENCE_LEVEL,
    MAX_DRAWDOWN_EXTREME_ANCHOR,
    STRESS_TEST_DISCLOSURE_THRESHOLD,
    WEIGHT_BETA,
    WEIGHT_SHARPE,
    WEIGHT_VAR,
    WEIGHT_MAX_DRAWDOWN,
    TRADING_DAYS_PER_YEAR,
)


# ─────────────────────────────────────────────
# Shared helper: daily returns
# ─────────────────────────────────────────────

def _daily_returns(prices: pd.Series) -> pd.Series:
    """
    Converts a price series into a simple daily returns series.

    Used as the common input for Beta, Sharpe, and VaR. Max Drawdown works
    directly on the price series instead (see max_drawdown section), since
    it needs the actual price path, not returns.

    Simple returns (not log returns) are used here for consistency with
    the historical-simulation VaR approach, which reads percentile cutoffs
    directly off this distribution.

    Args:
        prices: pd.Series of daily closing prices, chronologically ordered

    Returns:
        pd.Series of daily simple returns, one shorter than the input
        (the first day has no prior price to compare against, so pct_change()
        drops it via dropna()).
    """
    return prices.pct_change().dropna()


# ─────────────────────────────────────────────
# SUB-SIGNAL 1: Beta (weight: 30%)
# ─────────────────────────────────────────────

def _compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> dict | None:
    """
    Computes Beta via covariance/variance regression, then applies the
    Blume (1971) adjustment — the same methodology Bloomberg Terminal uses
    for its "Adjusted Beta". Raw Beta estimates tend to revert toward 1.0
    over time, so the adjustment pulls extreme raw estimates partway back
    toward the market average.

    Formula:
        raw_beta      = Cov(stock_returns, market_returns) / Var(market_returns)
        adjusted_beta = 0.67 * raw_beta + 0.33 * 1.0

    Args:
        stock_returns:  daily returns for the stock
        market_returns: daily returns for the market benchmark (^GSPC)

    Returns:
        dict with raw_beta, adjusted_beta, beta_score (-1.0 to +1.0), and detail
        or None if market_returns has no variance (degenerate case, should
        not happen in practice with real market data).
    """
    # Align on shared trading days — stock and benchmark calendars can
    # differ slightly (different exchange holidays, data gaps).
    aligned = pd.concat([stock_returns, market_returns], axis=1, join="inner")
    aligned.columns = ["stock", "market"]

    market_variance = aligned["market"].var()
    if market_variance == 0:
        return None

    covariance = aligned["stock"].cov(aligned["market"])
    raw_beta   = covariance / market_variance

    # Blume (1971) adjustment — pulls raw Beta toward 1.0
    adjusted_beta = BLUME_RAW_WEIGHT * raw_beta + BLUME_MARKET_WEIGHT * 1.0

    # Normalise to [-1, +1]: Beta=0 -> +1 (safest), Beta=1 -> 0 (neutral),
    # Beta=BETA_EXTREME_ANCHOR (2.0) -> -1 (riskiest)
    beta_score = max(-1.0, min(1.0, (1 - adjusted_beta) / (BETA_EXTREME_ANCHOR - 1)))

    return {
        "raw_beta":      round(raw_beta, 4),
        "adjusted_beta": round(adjusted_beta, 4),
        "beta_score":    round(beta_score, 4),
        "detail": (
            f"Raw Beta {round(raw_beta, 2)}, Blume-adjusted to {round(adjusted_beta, 2)} "
            f"(beta_score={round(beta_score, 2)})"
        ),
    }


# ─────────────────────────────────────────────
# SUB-SIGNAL 2: Sharpe Ratio (weight: 30%)
# ─────────────────────────────────────────────

def _compute_sharpe(stock_returns: pd.Series, risk_free_rate: float | None) -> dict:
    """
    Computes the Sharpe Ratio — risk-adjusted return efficiency.

    Formula:
        annualised_return     = mean(daily_returns) * 252
        annualised_volatility = std(daily_returns) * sqrt(252)
        sharpe = (annualised_return - risk_free_rate) / annualised_volatility

    Annualising uses 252 (approximate US trading days per year). Returns
    scale linearly with time (mean of independent sums), but standard
    deviation scales with the square root of time (variance of independent
    sums is additive, so std scales as sqrt(n)) — this is why volatility
    uses sqrt(252) while return uses a plain 252 multiplier.

    Args:
        stock_returns:  daily returns for the stock
        risk_free_rate: decimal annual rate (e.g. 0.045 for 4.5%), or None
                         if ^TNX could not be fetched — falls back to
                         RISK_FREE_RATE_FALLBACK in that case

    Returns:
        dict with annualised_return, volatility_annual, sharpe_ratio,
        sharpe_score (-1.0 to +1.0), and detail. Always returns a value
        (never None) — risk_free_rate has a fallback, and stock_returns
        is guaranteed non-empty by the sample-size guard in risk_signal().
    """
    if risk_free_rate is None:
        risk_free_rate = RISK_FREE_RATE_FALLBACK

    annualised_return     = stock_returns.mean() * TRADING_DAYS_PER_YEAR
    volatility_annual     = stock_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    if volatility_annual == 0:
        # No volatility at all (degenerate case, e.g. a flat/frozen price
        # series) — Sharpe is undefined. Treat as neutral rather than
        # dividing by zero.
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (annualised_return - risk_free_rate) / volatility_annual

    # Normalise via tanh: Sharpe=2.0 ("very good" per CFA convention)
    # maps to tanh(1.0) ≈ 0.76, leaving room above for exceptional cases.
    sharpe_score = np.tanh(sharpe_ratio / SHARPE_TANH_DIVISOR)

    return {
        "annualised_return": round(annualised_return, 4),
        "volatility_annual": round(volatility_annual, 4),
        "sharpe_ratio":      round(sharpe_ratio, 4),
        "sharpe_score":      round(float(sharpe_score), 4),
        "detail": (
            f"Annualised return {round(annualised_return*100, 2)}%, "
            f"volatility {round(volatility_annual*100, 2)}%, "
            f"risk-free rate {round(risk_free_rate*100, 2)}% "
            f"(sharpe_ratio={round(sharpe_ratio, 2)}, sharpe_score={round(float(sharpe_score), 2)})"
        ),
    }


# ─────────────────────────────────────────────
# SUB-SIGNAL 3: VaR — 95% Historical Simulation (weight: 20%)
# ─────────────────────────────────────────────

def _compute_var(stock_returns: pd.Series) -> dict:
    """
    Computes Value at Risk (95%) using the historical simulation method —
    directly reading the 5th percentile off the actual returns distribution,
    rather than assuming a normal distribution (real stock returns have
    "fat tails": extreme moves happen more often than a normal distribution
    predicts, so a parametric approach would understate tail risk).

    Interpretation: VaR_95 = -0.04 means "on 95% of trading days, the loss
    did not exceed 4%; only the worst 5% of days were worse than this."
    Note this describes *normal* conditions — it explicitly excludes true
    black-swan events, which is a known limitation of VaR as a metric.

    Args:
        stock_returns: daily returns for the stock

    Returns:
        dict with var_95 (negative float, e.g. -0.04), var_score (-1.0 to +1.0),
        and detail.
    """
    percentile = (1 - VAR_CONFIDENCE_LEVEL) * 100  # 95% confidence -> 5th percentile
    var_95 = np.percentile(stock_returns, percentile)

    # Normalise: VaR=0 -> +1 (safest), VaR=VAR_EXTREME_ANCHOR (-10%) -> -1 (riskiest)
    var_score = max(-1.0, min(1.0, 1 + var_95 / abs(VAR_EXTREME_ANCHOR)))

    return {
        "var_95":    round(float(var_95), 4),
        "var_score": round(float(var_score), 4),
        "detail": (
            f"95% VaR (1-day) is {round(var_95 * 100, 2)}% — on 95% of trading "
            f"days, losses did not exceed this (var_score={round(var_score, 2)}). "
            f"This reflects normal market conditions and does not capture "
            f"black-swan events."
        ),
    }


# ─────────────────────────────────────────────
# SUB-SIGNAL 4: Max Drawdown (weight: 20%)
# ─────────────────────────────────────────────

def _compute_max_drawdown(prices: pd.Series) -> dict:
    """
    Computes the maximum peak-to-trough decline over the price history.

    Unlike Beta/Sharpe/VaR, this works directly on prices (not returns) and
    uses the *rolling* historical peak at each point in time — not the
    single highest price in the whole window. This matters: drawdown
    simulates what a real investor could have experienced (buying at some
    point, then watching the price fall from whatever the highest point
    was *up to that day*), not an abstract "highest minus lowest" spread
    that could span a peak occurring *after* the trough.

    Formula:
        drawdown(t)      = (price(t) - rolling_peak(t)) / rolling_peak(t)
        max_drawdown     = min(drawdown(t)) over all t

    Limitation (disclosed, not corrected): this is a rolling-window
    calculation over the fetched price history only (~2 years). It does
    not include a comparison against major historical crises (e.g. 2000
    dot-com crash, 2008 financial crisis) — that kind of scenario-based
    stress testing is a separate exercise, planned for the Backtesting
    Engine (Layer 2 Step 10), not this signal. If the window happened to
    avoid any real downturn, this metric may understate true risk — the
    detail field flags this explicitly rather than silently reporting an
    optimistic number.

    Args:
        prices: pd.Series of daily closing prices, chronologically ordered

    Returns:
        dict with max_drawdown (negative float, e.g. -0.24), drawdown_score
        (-1.0 to +1.0), stress_tested (bool), and detail.
    """
    rolling_peak = prices.cummax()
    drawdown_series = (prices - rolling_peak) / rolling_peak
    max_drawdown = drawdown_series.min()

    # Normalise: MaxDD=0 -> +1 (safest), MaxDD=MAX_DRAWDOWN_EXTREME_ANCHOR (-60%) -> -1
    drawdown_score = max(-1.0, min(1.0, 1 + max_drawdown / abs(MAX_DRAWDOWN_EXTREME_ANCHOR)))

    # Disclosure flag: did this window experience a "meaningful" drawdown,
    # or did it get lucky and avoid one? This is informational only —
    # it does not change drawdown_score.
    stress_tested = max_drawdown <= STRESS_TEST_DISCLOSURE_THRESHOLD

    if stress_tested:
        stress_note = (
            f"This window included a decline of at least "
            f"{abs(STRESS_TEST_DISCLOSURE_THRESHOLD)*100:.0f}%, providing some "
            f"evidence of how the stock behaves under stress."
        )
    else:
        stress_note = (
            f"This window did not include a decline of "
            f"{abs(STRESS_TEST_DISCLOSURE_THRESHOLD)*100:.0f}% or more — the stock "
            f"may not have been tested by a real downturn recently, so this "
            f"metric may understate risk in a genuine market crisis."
        )

    return {
        "max_drawdown":    round(float(max_drawdown), 4),
        "drawdown_score":  round(float(drawdown_score), 4),
        "stress_tested":   bool(stress_tested),
        "detail": (
            f"Maximum peak-to-trough decline over the observed window was "
            f"{round(max_drawdown * 100, 2)}% (drawdown_score={round(drawdown_score, 2)}). "
            f"{stress_note}"
        ),
    }


# ─────────────────────────────────────────────
# MAIN FUNCTION: risk_signal
# ─────────────────────────────────────────────

def risk_signal(market_data: dict, risk_inputs: dict | None) -> dict | None:
    """
    Compute a risk signal from price history data.

    Pure function — takes already-fetched data, performs no I/O. Data
    fetching is done separately by market_data.get_risk_inputs(), called
    by quant_engine.py before this function runs (same pattern as
    valuation_signal.py and momentum_signal.py).

    Degradation strategy (two levels, matching the actual failure modes):
        1. Stock's own price history missing/too short -> returns None.
           None of the four metrics can be computed without it.
        2. Market benchmark (^GSPC) missing -> Beta is skipped, the other
           three metrics (Sharpe, VaR, Max Drawdown) are computed normally,
           and the composite score dynamically re-normalises weights over
           the remaining signals (same approach as momentum_signal.py) —
           this is the standard way index/portfolio weighting schemes
           handle a missing constituent, rather than assuming a neutral
           value of 0 for the missing piece.

    Args:
        market_data: dict from get_stock_snapshot() — not used for
                     calculations directly, but accepted for interface
                     consistency with valuation_signal()/momentum_signal(),
                     and to access the ticker for logging/detail messages.
        risk_inputs: dict from get_risk_inputs(), expected keys:
            - stock_prices:   pd.Series of daily closing prices
            - market_prices:  pd.Series | None (^GSPC)
            - risk_free_rate: float | None (^TNX as decimal)

    Returns:
        dict with keys:
            - risk_score:        float (-1.0 to +1.0) or None
            - risk_level:        str — "low" / "medium" / "high"
            - low_confidence:    bool — True if sample size is below
                                  LOW_CONFIDENCE_THRESHOLD (~1 year)
            - beta:               dict | None (from _compute_beta, None if
                                  market_prices unavailable)
            - sharpe:             dict (from _compute_sharpe)
            - var:                dict (from _compute_var)
            - max_drawdown:       dict (from _compute_max_drawdown)
            - detail:             str — plain English explanation
        or None if the stock's own price history is missing or too short
        (Tier equivalent to valuation_signal's Tier 4 — insufficient data).
    """
    if risk_inputs is None:
        return None

    stock_prices = risk_inputs.get("stock_prices")
    if stock_prices is None or len(stock_prices) < MIN_TRADING_DAYS:
        return None

    stock_returns = _daily_returns(stock_prices)

    low_confidence = len(stock_prices) < LOW_CONFIDENCE_THRESHOLD

    # ── Sub-signal 2, 3, 4: always computable from stock's own data ──────
    sharpe_result = _compute_sharpe(stock_returns, risk_inputs.get("risk_free_rate"))
    var_result    = _compute_var(stock_returns)
    mdd_result    = _compute_max_drawdown(stock_prices)

    # ── Sub-signal 1: Beta, only if market benchmark is available ────────
    market_prices = risk_inputs.get("market_prices")
    beta_result = None
    if market_prices is not None and len(market_prices) >= MIN_TRADING_DAYS:
        market_returns = _daily_returns(market_prices)
        beta_result = _compute_beta(stock_returns, market_returns)

    # ── Composite score: dynamic re-normalisation over available signals ─
    # Matches momentum_signal.py's approach — if a component is missing,
    # redistribute its weight proportionally over what's available, rather
    # than assuming a neutral score of 0 for the missing piece.
    available = {
        "beta":     (beta_result["beta_score"] if beta_result else None, WEIGHT_BETA),
        "sharpe":   (sharpe_result["sharpe_score"], WEIGHT_SHARPE),
        "var":      (var_result["var_score"], WEIGHT_VAR),
        "drawdown": (mdd_result["drawdown_score"], WEIGHT_MAX_DRAWDOWN),
    }

    total_weight = sum(w for _, (s, w) in available.items() if s is not None)

    if total_weight == 0:
        return None  # should not happen in practice — guarded above

    risk_score = sum(
        s * w for _, (s, w) in available.items() if s is not None
    ) / total_weight

    risk_score = round(max(-1.0, min(1.0, risk_score)), 4)

    # ── Label ──────────────────────────────────────────────────────────
    if risk_score > 0.2:
        risk_level = "low"
    elif risk_score < -0.2:
        risk_level = "high"
    else:
        risk_level = "medium"

    # ── Detail: assembled from sub-signal details, plus missing-data note ─
    detail_parts = []
    if beta_result:
        detail_parts.append(beta_result["detail"])
    else:
        detail_parts.append("Beta could not be computed — market benchmark data unavailable.")
    detail_parts.append(sharpe_result["detail"])
    detail_parts.append(var_result["detail"])
    detail_parts.append(mdd_result["detail"])

    # Always disclose the observation window length, regardless of
    # confidence level — users should know the time horizon these metrics
    # are based on even when the sample size is fully adequate, not only
    # when it falls short.
    years_approx = round(len(stock_prices) / TRADING_DAYS_PER_YEAR, 1)
    detail_parts.append(
        f"These metrics are based on {len(stock_prices)} trading days "
        f"(approximately {years_approx} years) of price history."
    )

    if low_confidence:
        detail_parts.append(
            f"Note: this is less than a full {LOW_CONFIDENCE_THRESHOLD}-day "
            f"(~1 year) window — confidence in these estimates is reduced."
        )

    return {
        "risk_score":      risk_score,
        "risk_level":      risk_level,
        "low_confidence":  low_confidence,
        "beta":            beta_result,
        "sharpe":          sharpe_result,
        "var":             var_result,
        "max_drawdown":    mdd_result,
        "detail":          " | ".join(detail_parts),
    }
