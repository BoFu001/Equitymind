"""
src/quant/risk_signal_config.py

Configuration constants for the Risk Signal Engine.
Keeping these separate from risk_signal.py mirrors the peer_benchmarks.json
pattern used by valuation_signal.py — numeric assumptions live in one place,
separate from calculation logic, so they can be tuned without touching code.

Each constant below is either:
    (a) a documented industry convention (cited), or
    (b) a reasoned reference point anchored to a real historical benchmark
        (cited), not an arbitrary guess.
"""

# ─────────────────────────────────────────────
# Sample size thresholds
# ─────────────────────────────────────────────
# Below this many trading days, covariance/variance-based estimates
# (Beta, Sharpe) are dominated by noise rather than signal — the whole
# Risk Signal returns None rather than reporting an unreliable number.
MIN_TRADING_DAYS = 60

# Below this many trading days (but above MIN_TRADING_DAYS), the signal
# is computed but flagged as low_confidence — less than a full 2-year
# window (~504 days) worth of data.
LOW_CONFIDENCE_THRESHOLD = 252  # ~1 year

# ─────────────────────────────────────────────
# Beta — Blume Adjustment (Blume, 1971)
# Raw Beta estimates from historical regression tend to revert toward 1.0
# over time. Bloomberg Terminal applies this same adjustment as its
# standard "Adjusted Beta" methodology.
# ─────────────────────────────────────────────
BLUME_RAW_WEIGHT    = 0.67
BLUME_MARKET_WEIGHT = 0.33  # pulled toward Beta = 1.0

# Normalisation anchor: adjusted Beta of 2.0 maps to score = -1.0 (max risk)
# Adjusted Beta is compressed toward 1.0 by design, so 2.0 remains a
# generously wide bound for the clip.
BETA_EXTREME_ANCHOR = 2.0

# ─────────────────────────────────────────────
# Sharpe Ratio
# ─────────────────────────────────────────────
# tanh divisor: Sharpe of 2.0 ("very good" per CFA convention) maps to
# tanh(1.0) ≈ 0.76, leaving room above it for exceptional (rare) cases.
SHARPE_TANH_DIVISOR = 2.0

# Fallback risk-free rate if ^TNX fetch fails. Approximate recent
# 10-year Treasury yield level; only used when live data is unavailable.
RISK_FREE_RATE_FALLBACK = 0.04

# ─────────────────────────────────────────────
# VaR (95%, historical simulation method)
# ─────────────────────────────────────────────
# Normalisation anchor: a single-day VaR of -10% maps to score = -1.0.
# Reference point: US single-stock circuit breakers (LULD) typically
# trigger in the 5-10% range for liquid securities.
VAR_EXTREME_ANCHOR = -0.10
VAR_CONFIDENCE_LEVEL = 0.95  # 95% historical VaR — standard retail-facing convention

# ─────────────────────────────────────────────
# Max Drawdown
# ─────────────────────────────────────────────
# Normalisation anchor: a max drawdown of -60% maps to score = -1.0.
# Reference point: S&P 500 fell ~57% peak-to-trough in the 2008
# financial crisis; -60% represents a "near-historic-crisis" magnitude.
MAX_DRAWDOWN_EXTREME_ANCHOR = -0.60

# Threshold below which the 2-year window is flagged as NOT having
# experienced a meaningful drawdown — i.e. the risk score may understate
# true risk because this window got lucky and avoided a real downturn.
# This is a disclosure flag, not a scoring input.
STRESS_TEST_DISCLOSURE_THRESHOLD = -0.15

# ─────────────────────────────────────────────
# Composite score weights
# Beta and Sharpe carry more weight as they reflect systematic exposure
# and risk-adjusted return efficiency — the metrics professional
# investors check first. VaR and Max Drawdown provide supplementary
# tail-risk confirmation.
# ─────────────────────────────────────────────
WEIGHT_BETA         = 0.30
WEIGHT_SHARPE       = 0.30
WEIGHT_VAR          = 0.20
WEIGHT_MAX_DRAWDOWN = 0.20

# ─────────────────────────────────────────────
# Trading days per year (US market convention)
# ─────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
