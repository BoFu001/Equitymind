"""
src/quant/quality_signal_config.py

Configuration constants for the Quality Signal Engine (Piotroski F-Score).
"""

# ─────────────────────────────────────────────
# F-Score to composite score mapping
# ─────────────────────────────────────────────
# F-Score ranges from 0 to 9 (number of the 9 signals satisfied).
# Normalise to [-1, +1]: F-Score=0 -> -1 (worst), F-Score=9 -> +1 (best),
# F-Score=4.5 -> 0 (neutral midpoint).
MAX_F_SCORE = 9

# ─────────────────────────────────────────────
# Label thresholds (consistent with valuation_signal.py / momentum_signal.py style)
# ─────────────────────────────────────────────
# Piotroski's original research treats F-Score 8-9 as "high quality" and
# 0-2 as "low quality" (most likely to be value traps if also cheap).
HIGH_QUALITY_THRESHOLD = 7   # score >= 7 -> "high"
LOW_QUALITY_THRESHOLD  = 3   # score <= 3 -> "low"
