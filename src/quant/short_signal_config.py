"""
src/quant/short_signal_config.py

Configuration constants for the Short Signal Engine.

Only two evidence-graded numbers are kept here (2026-07-31 review):
    - DAYS_TO_COVER_ELEVATED_THRESHOLD: industry-practice convention,
      corroborated by 3 independent sources (Britannica Money,
      WhaleQuant, Deepvue) — not peer-reviewed, but real and cited.
    - Everything else previously here (a 2%/5% Short Interest label
      system, a 20%-anchored -1/+1 score, a 100% historic-extreme
      flag) was removed — each depended on a self-selected number
      with no academic or industry source. Only short_reader.py's raw
      numbers are reported for those metrics; no derived label/score.
"""

DAYS_TO_COVER_ELEVATED_THRESHOLD = 5.0   # >= 5 days -> "elevated" (industry convention, not academic)
