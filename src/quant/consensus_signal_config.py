"""
src/quant/consensus_signal_config.py

Configuration constants for the Consensus Signal Engine (analyst sentiment).
"""

# NOTE: no weights here. Recommendation and upside answer different
# questions — where the rating stands now, and how far the target price
# sits above the current one — and are returned independently rather
# than blended into one number.

# ─────────────────────────────────────────────
# Confidence / disclosure thresholds
# ─────────────────────────────────────────────
MIN_ANALYST_COUNT = 5        # below this, flag low sample size
WIDE_TARGET_RANGE_RATIO = 3  # if target_high > 3x target_low, flag high dispersion
