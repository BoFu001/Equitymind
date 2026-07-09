"""
src/quant/consensus_signal_config.py

Configuration constants for the Consensus Signal Engine (analyst sentiment).
"""

# ─────────────────────────────────────────────
# Sub-signal weights — must sum to 1.0
# Recommendation carries the most weight because it has the largest,
# most stable sample size (all covering analysts, every period).
# Upside and Trend are weighted equally as secondary, more volatile inputs.
# ─────────────────────────────────────────────
WEIGHT_RECOMMENDATION = 0.4
WEIGHT_UPSIDE         = 0.3
WEIGHT_TREND          = 0.3

# ─────────────────────────────────────────────
# Calibration anchors
# ─────────────────────────────────────────────
# Upside: 50% implied upside -> score of +1.0 (same anchor used when this
# logic lived inside valuation_signal.py, before Consensus Signal existed)
UPSIDE_CAP_PCT = 0.50

# Trend: a 0.5-point swing in the weighted recommendation scale (1-5)
# over the observed window -> trend_score of +-1.0
TREND_CAP_POINTS = 0.5

# ─────────────────────────────────────────────
# Confidence / disclosure thresholds
# ─────────────────────────────────────────────
MIN_ANALYST_COUNT = 5        # below this, flag low sample size
WIDE_TARGET_RANGE_RATIO = 3  # if target_high > 3x target_low, flag high dispersion

# ─────────────────────────────────────────────
# Label thresholds (consistent with other signal engines' style)
# ─────────────────────────────────────────────
BULLISH_THRESHOLD  = 0.3
BEARISH_THRESHOLD  = -0.3
