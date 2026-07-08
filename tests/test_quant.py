"""
Unit tests for Layer 2 Quantitative Signal Engines.

All tests use mock market data — no real API calls are made.
Benchmark values are loaded dynamically from
src/quant/data/peer_benchmarks.json so tests remain valid after
quarterly benchmark updates via scripts/update_peer_groups.py.
"""


import pytest
from src.quant.valuation_signal import (
    valuation_signal,
    DEFAULT_PE,
    DEFAULT_PB,
    DEFAULT_PS,
    BENCHMARKS_STALE,
)


# ─────────────────────────────────────────────
# Helper — build mock market data dicts
# ─────────────────────────────────────────────

def make_market_data(**kwargs) -> dict:
    """
    Return a minimal market data dict for testing.
    All fields mirror those returned by get_market_data_tool.
    Override any field via keyword arguments.
    """
    defaults = {
        "ticker":         "AAPL",
        "company_name":   "Test Corp",
        "sector":         "Technology",
        "current_price":  100.0,
        "pe_ratio":       float(DEFAULT_PE),
        "price_to_book":  float(DEFAULT_PB),
        "price_to_sales": float(DEFAULT_PS),
    }
    defaults.update(kwargs)
    return defaults


# ─────────────────────────────────────────────
# Tier 1 — P/E + P/B (most reliable)
# ─────────────────────────────────────────────

class TestTier1:

    def test_fairly_valued(self):
        """P/E and P/B both equal peer average → score near zero."""
        data   = make_market_data()
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "pe_pb"
        assert result["reference_only"]  is False
        assert -0.2 <= result["valuation_score"] <= 0.2
        assert result["valuation_label"] == "fairly valued"

    def test_undervalued(self):
        """P/E and P/B well below peer average → undervalued."""
        data   = make_market_data(pe_ratio=5.0, price_to_book=1.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "pe_pb"
        assert result["valuation_score"] >  0.2
        assert result["valuation_label"] == "undervalued"

    def test_overvalued(self):
        """P/E and P/B well above peer average → overvalued."""
        data   = make_market_data(pe_ratio=300.0, price_to_book=100.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "pe_pb"
        assert result["valuation_score"] <  -0.2
        assert result["valuation_label"] == "overvalued"

    def test_detail_string_non_empty(self):
        """Result must include a non-empty plain-English detail string."""
        result = valuation_signal(make_market_data())
        assert result is not None
        assert "detail" in result
        assert len(result["detail"]) > 0

    def test_pe_vs_peers_string_present(self):
        """pe_vs_peers should describe the comparison as a readable string."""
        result = valuation_signal(make_market_data())
        assert result is not None
        assert result["pe_vs_peers"] is not None
        assert "vs peer avg" in result["pe_vs_peers"]

    def test_stale_benchmark_is_bool(self):
        """stale_benchmark must be a boolean."""
        result = valuation_signal(make_market_data())
        assert result is not None
        assert isinstance(result["stale_benchmark"], bool)

    def test_missing_pb_degrades_to_pe_only(self):
        """Missing P/B should degrade gracefully to a P/E-only score."""
        data   = make_market_data(price_to_book=None)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"] == "pe_pb"
        assert "P/B data unavailable" in result["detail"]

    def test_zero_pb_treated_as_missing(self):
        """A P/B of exactly 0 should be treated as missing, not a real ratio."""
        data   = make_market_data(price_to_book=0)
        result = valuation_signal(data)
        assert result is not None
        assert "P/B data unavailable" in result["detail"]


# ─────────────────────────────────────────────
# Tier 2 — P/S only (loss-making companies, reference only)
# ─────────────────────────────────────────────

class TestTier2:

    def test_loss_making_company_uses_ps(self):
        """Negative P/E with available P/S → Tier 2 (reference only)."""
        data   = make_market_data(pe_ratio=-10.0, price_to_book=None, price_to_sales=8.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "ps_only"
        assert result["reference_only"]  is True
        assert result["valuation_label"] == "reference only"

    def test_no_pe_falls_to_tier2(self):
        """No P/E data at all → Tier 2 if P/S is available."""
        data   = make_market_data(pe_ratio=None, price_to_book=None, price_to_sales=5.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]         == "ps_only"
        assert result["reference_only"] is True

    def test_detail_warns_reference_only(self):
        """Tier 2 detail must explicitly warn the score is for reference only."""
        data   = make_market_data(pe_ratio=None, price_to_book=None, price_to_sales=5.0)
        result = valuation_signal(data)
        assert result is not None
        assert "reference only" in result["detail"].lower()

    def test_pe_vs_peers_is_none(self):
        """Tier 2 must not populate pe_vs_peers (P/E is not used)."""
        data   = make_market_data(pe_ratio=None, price_to_book=None, price_to_sales=6.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["pe_vs_peers"] is None

    def test_stale_benchmark_present(self):
        """Tier 2 result must include stale_benchmark field."""
        data   = make_market_data(pe_ratio=None, price_to_book=None, price_to_sales=5.0)
        result = valuation_signal(data)
        assert result is not None
        assert "stale_benchmark" in result
        assert isinstance(result["stale_benchmark"], bool)


# ─────────────────────────────────────────────
# Tier 3 — No usable data → returns None
# ─────────────────────────────────────────────

class TestTier3:

    def test_no_data_returns_none(self):
        """No P/E, P/B, or P/S → must return None to prevent hallucination."""
        data   = make_market_data(pe_ratio=None, price_to_book=None, price_to_sales=None)
        result = valuation_signal(data)
        assert result is None

    def test_zero_values_return_none(self):
        """Zero P/E and P/S must be treated as missing data."""
        data   = make_market_data(pe_ratio=0, price_to_book=0, price_to_sales=0)
        result = valuation_signal(data)
        assert result is None


# ─────────────────────────────────────────────
# Score bounds — extreme inputs must never exceed ±1.0
# ─────────────────────────────────────────────

class TestScoreBounds:

    def test_score_clamped_on_extreme_cheap(self):
        """Extremely cheap stock must not produce score > +1.0."""
        data   = make_market_data(pe_ratio=0.5, price_to_book=0.1)
        result = valuation_signal(data)
        assert result is not None
        assert result["valuation_score"] <= 1.0

    def test_score_clamped_on_extreme_expensive(self):
        """Extremely expensive stock must not produce score < -1.0."""
        data   = make_market_data(pe_ratio=5000.0, price_to_book=1000.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["valuation_score"] >= -1.0


# ─────────────────────────────────────────────
# JSON benchmark loading
# ─────────────────────────────────────────────

class TestBenchmarkLoading:

    def test_default_pe_is_positive_number(self):
        """DEFAULT_PE must be a positive number loaded from JSON."""
        assert isinstance(DEFAULT_PE, (int, float))
        assert DEFAULT_PE > 0

    def test_default_pb_is_positive_number(self):
        """DEFAULT_PB must be a positive number loaded from JSON."""
        assert isinstance(DEFAULT_PB, (int, float))
        assert DEFAULT_PB > 0

    def test_default_ps_is_positive_number(self):
        """DEFAULT_PS must be a positive number."""
        assert isinstance(DEFAULT_PS, (int, float))
        assert DEFAULT_PS > 0

    def test_unknown_ticker_uses_default_pe(self):
        """Unknown ticker must fall back to DEFAULT_PE without crashing."""
        data   = make_market_data(ticker="UNKNOWN")
        result = valuation_signal(data)
        assert result is not None
        assert str(int(DEFAULT_PE)) in result["pe_vs_peers"]

    def test_benchmarks_stale_is_bool(self):
        """BENCHMARKS_STALE must be a boolean."""
        assert isinstance(BENCHMARKS_STALE, bool)


# Helpers — mock market data for momentum
# ─────────────────────────────────────────────

from src.quant.momentum_signal import momentum_signal

def make_momentum_data(**kwargs) -> dict:
    """Create a minimal mock market data dict for momentum testing."""
    defaults = {
        "rsi":           50.0,
        "macd":          0.0,
        "macd_signal":   0.0,
        "current_price": 100.0,
        "sma_50":        100.0,
        "sma_200":       100.0,
        "52w_high":      120.0,
        "52w_low":       80.0,
    }
    defaults.update(kwargs)
    return defaults


# ─────────────────────────────────────────────
# Momentum Signal — Neutral baseline
# ─────────────────────────────────────────────

class TestMomentumNeutral:

    def test_neutral_baseline(self):
        """RSI=50, MACD=Signal, price=SMA50=SMA200, mid 52w range → neutral."""
        result = momentum_signal(make_momentum_data())
        assert result is not None
        assert result["momentum_label"] == "neutral"
        assert -0.2 <= result["momentum_score"] <= 0.2

    def test_returns_all_fields(self):
        """Result must include all required fields."""
        result = momentum_signal(make_momentum_data())
        assert result is not None
        for field in ["momentum_score", "momentum_label", "rsi_score", "macd_score", "price_score", "detail"]:
            assert field in result

# ─────────────────────────────────────────────
# Risk Signal Engine
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd

from src.quant.risk_signal import risk_signal
from src.quant.risk_signal_config import (
    MIN_TRADING_DAYS,
    LOW_CONFIDENCE_THRESHOLD,
    BLUME_RAW_WEIGHT,
    BLUME_MARKET_WEIGHT,
)


def make_price_series(returns: np.ndarray, start_price: float = 100.0) -> pd.Series:
    """
    Build a pd.Series of prices from an array of daily returns.
    Deterministic — no randomness — so tests can hand-calculate expected
    Beta/Sharpe/VaR/MaxDD rather than just checking "did it run".
    """
    prices = start_price * np.cumprod(1 + returns)
    dates  = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=dates)


def make_risk_inputs(
    n_days: int = 504,
    beta_true: float = 1.5,
    market_seed: int = 42,
    risk_free_rate: float = 0.04,
    include_market: bool = True,
) -> dict:
    """
    Build a mock risk_inputs dict with a known, reproducible relationship
    between stock and market returns — stock_returns = beta_true * market_returns
    exactly (no noise), so the computed raw Beta should equal beta_true.
    """
    rng = np.random.default_rng(market_seed)
    market_returns = rng.normal(loc=0.0004, scale=0.01, size=n_days)
    stock_returns  = beta_true * market_returns  # exact linear relationship

    stock_prices  = make_price_series(stock_returns)
    market_prices = make_price_series(market_returns) if include_market else None

    return {
        "stock_prices":   stock_prices,
        "market_prices":  market_prices,
        "risk_free_rate": risk_free_rate,
    }


class TestRiskSignalNormalCase:
    def test_beta_matches_known_relationship(self):
        """
        Stock returns are constructed as an exact multiple of market returns
        (beta_true=1.5), so raw_beta should recover ~1.5, and adjusted_beta
        should reflect the Blume adjustment formula exactly.
        """
        market_data = {"ticker": "TEST"}
        risk_inputs = make_risk_inputs(n_days=504, beta_true=1.5)

        result = risk_signal(market_data, risk_inputs)

        assert result is not None
        assert result["beta"] is not None
        assert result["beta"]["raw_beta"] == pytest.approx(1.5, abs=0.01)

        expected_adjusted = BLUME_RAW_WEIGHT * 1.5 + BLUME_MARKET_WEIGHT * 1.0
        assert result["beta"]["adjusted_beta"] == pytest.approx(expected_adjusted, abs=0.01)

    def test_composite_score_in_range(self):
        """Composite risk_score must always fall within [-1.0, +1.0]."""
        market_data = {"ticker": "TEST"}
        risk_inputs = make_risk_inputs(n_days=504, beta_true=1.0)

        result = risk_signal(market_data, risk_inputs)

        assert result is not None
        assert -1.0 <= result["risk_score"] <= 1.0
        assert result["risk_level"] in ("low", "medium", "high")

    def test_low_confidence_false_with_full_window(self):
        """A full 504-day (2-year) window should not be flagged low_confidence."""
        market_data = {"ticker": "TEST"}
        risk_inputs = make_risk_inputs(n_days=504)

        result = risk_signal(market_data, risk_inputs)

        assert result["low_confidence"] is False


class TestRiskSignalMissingMarketData:
    def test_beta_skipped_when_market_prices_missing(self):
        """
        ^GSPC unavailable -> Beta should be skipped (None), but Sharpe,
        VaR, and Max Drawdown should still compute normally, and the
        composite score should re-normalise over the remaining three.
        """
        market_data = {"ticker": "TEST"}
        risk_inputs = make_risk_inputs(n_days=504, include_market=False)

        result = risk_signal(market_data, risk_inputs)

        assert result is not None
        assert result["beta"] is None
        assert result["sharpe"] is not None
        assert result["var"] is not None
        assert result["max_drawdown"] is not None
        assert "market benchmark data unavailable" in result["detail"]

    def test_risk_free_rate_missing_uses_fallback(self):
        """
        ^TNX unavailable (risk_free_rate=None) should not crash — Sharpe
        falls back to RISK_FREE_RATE_FALLBACK instead of failing.
        """
        market_data = {"ticker": "TEST"}
        risk_inputs = make_risk_inputs(n_days=504, risk_free_rate=None)

        result = risk_signal(market_data, risk_inputs)

        assert result is not None
        assert result["sharpe"] is not None


class TestRiskSignalInsufficientData:
    def test_returns_none_when_risk_inputs_is_none(self):
        """get_risk_inputs() itself failed (e.g. network error) -> None."""
        market_data = {"ticker": "TEST"}
        result = risk_signal(market_data, None)
        assert result is None

    def test_returns_none_below_min_trading_days(self):
        """Fewer than MIN_TRADING_DAYS (60) -> too noisy to report, return None."""
        market_data = {"ticker": "TEST"}
        risk_inputs = make_risk_inputs(n_days=MIN_TRADING_DAYS - 1)

        result = risk_signal(market_data, risk_inputs)

        assert result is None

    def test_low_confidence_true_between_thresholds(self):
        """
        Between MIN_TRADING_DAYS (60) and LOW_CONFIDENCE_THRESHOLD (252):
        computable, but flagged as low_confidence.
        """
        market_data = {"ticker": "TEST"}
        n_days = (MIN_TRADING_DAYS + LOW_CONFIDENCE_THRESHOLD) // 2
        risk_inputs = make_risk_inputs(n_days=n_days)

        result = risk_signal(market_data, risk_inputs)

        assert result is not None
        assert result["low_confidence"] is True


# ─────────────────────────────────────────────
# Quality Signal Engine (Piotroski F-Score)
# ─────────────────────────────────────────────
from src.quant.quality_signal import quality_signal
from src.quant.quality_signal_config import (
    MAX_F_SCORE,
    HIGH_QUALITY_THRESHOLD,
    LOW_QUALITY_THRESHOLD,
)


def make_quality_inputs(**overrides) -> dict:
    """
    Build a mock quality_inputs dict with "current" and "prior" periods.
    Defaults represent a company satisfying all 9 Piotroski signals —
    override individual fields to test specific failure/edge conditions.
    """
    current = {
        "net_income":          600_000_000.0,
        "operating_cash_flow": 700_000_000.0,
        "total_assets":        5_500_000_000.0,
        "gross_profit":        1_512_000_000.0,   # 42% margin on 3.6B revenue
        "total_revenue":       3_600_000_000.0,
        "long_term_debt":      1_200_000_000.0,
        "current_assets":      900_000_000.0,
        "current_liabilities": 500_000_000.0,
        "shares_outstanding":  1_000_000_000.0,
    }
    prior = {
        "net_income":          500_000_000.0,
        "operating_cash_flow": 400_000_000.0,
        "total_assets":        5_000_000_000.0,
        "gross_profit":        1_200_000_000.0,   # 40% margin on 3.0B revenue
        "total_revenue":       3_000_000_000.0,
        "long_term_debt":      1_500_000_000.0,
        "current_assets":      800_000_000.0,
        "current_liabilities": 600_000_000.0,
        "shares_outstanding":  1_000_000_000.0,
    }
    current.update(overrides.get("current", {}))
    prior.update(overrides.get("prior", {}))
    return {"current": current, "prior": prior}


class TestQualitySignalNormalCase:
    def test_all_nine_signals_pass(self):
        """
        A textbook-healthy company (City Brew Coffee style numbers):
        profitable, cash-flow-backed earnings, deleveraging, improving
        liquidity, no dilution, expanding margins and turnover.
        All 9 signals should pass -> F-Score 9/9, quality_score +1.0.
        """
        inputs = make_quality_inputs()
        result = quality_signal(inputs)

        assert result is not None
        assert result["f_score_raw"] == 9
        assert result["signals_evaluated"] == 9
        assert result["f_score"] == pytest.approx(9.0)
        assert result["quality_score"] == pytest.approx(1.0)
        assert result["quality_label"] == "high"

    def test_composite_score_in_range(self):
        """quality_score must always fall within [-1.0, +1.0]."""
        inputs = make_quality_inputs()
        result = quality_signal(inputs)

        assert -1.0 <= result["quality_score"] <= 1.0

    def test_all_nine_signals_fail(self):
        """
        A deteriorating company: losses, negative cash flow, rising
        leverage, worsening liquidity, dilution, shrinking margins —
        should score F-Score 0/9, quality_score -1.0, label "low".
        """
        inputs = make_quality_inputs(
            current={
                "net_income":          -100_000_000.0,
                "operating_cash_flow": -150_000_000.0,   # more negative than net income -> fails signal 4 too
                "total_assets":        5_000_000_000.0,
                "gross_profit":        1_000_000_000.0,   # 33% margin on 3.0B revenue (down from 40%)
                "total_revenue":       3_000_000_000.0,
                "long_term_debt":      2_000_000_000.0,   # ratio up
                "current_assets":      500_000_000.0,
                "current_liabilities": 700_000_000.0,     # ratio down
                "shares_outstanding":  1_200_000_000.0,   # diluted
            }
        )
        result = quality_signal(inputs)

        assert result is not None
        assert result["f_score_raw"] == 0
        assert result["quality_score"] == pytest.approx(-1.0)
        assert result["quality_label"] == "low"


class TestQualitySignalPartialData:
    def test_missing_long_term_debt_skips_leverage_signal_only(self):
        """
        If long_term_debt is None in either period (not 0 — genuinely
        missing), only the leverage signal should be skipped; the other
        8 should still be evaluated and the score rescaled accordingly.
        """
        inputs = make_quality_inputs()
        inputs["current"]["long_term_debt"] = None

        result = quality_signal(inputs)

        assert result is not None
        assert result["signals_evaluated"] == 8
        assert result["breakdown"]["leverage_decreasing"]["score"] is None
        # Remaining 8 signals should all still pass with default mock data
        assert result["f_score_raw"] == 8

    def test_zero_long_term_debt_is_not_missing(self):
        """
        A company with genuinely zero long-term debt (e.g. RDDT, PLTR)
        should NOT be treated as missing data — 0/0 ratio comparison
        (0% vs 0%) correctly fails the "decreasing" signal since it
        didn't decrease, but the signal IS evaluable.
        """
        inputs = make_quality_inputs(
            current={"long_term_debt": 0.0},
            prior={"long_term_debt": 0.0},
        )
        result = quality_signal(inputs)

        assert result is not None
        assert result["signals_evaluated"] == 9
        assert result["breakdown"]["leverage_decreasing"]["score"] == 0  # 0% is not < 0%


class TestQualitySignalShareDilutionEdgeCase:
    def test_negligible_dilution_still_fails_but_notes_magnitude(self):
        """
        Piotroski's original definition is strict: any increase fails,
        however small. The detail should note the tiny magnitude without
        speculating on cause (mirrors the real MSFT case: 7.434158B vs
        7.434139B shares, a 0.00027% increase).
        """
        inputs = make_quality_inputs(
            current={"shares_outstanding": 7_434_158_655.0},
            prior={"shares_outstanding": 7_434_138_859.0},
        )
        result = quality_signal(inputs)

        breakdown = result["breakdown"]["no_share_dilution"]
        assert breakdown["score"] == 0
        assert "very small magnitude" in breakdown["detail"]

    def test_buyback_reduces_shares_passes(self):
        """A share count reduction via buyback should pass this signal."""
        inputs = make_quality_inputs(
            current={"shares_outstanding": 900_000_000.0},
            prior={"shares_outstanding": 1_000_000_000.0},
        )
        result = quality_signal(inputs)

        assert result["breakdown"]["no_share_dilution"]["score"] == 1


class TestQualitySignalInsufficientData:
    def test_returns_none_when_quality_inputs_is_none(self):
        """get_quality_inputs() itself failed (e.g. <2 fiscal years) -> None."""
        result = quality_signal(None)
        assert result is None

    def test_returns_none_when_no_signals_evaluable(self):
        """
        Every required field missing across both periods -> no signal can
        be evaluated -> None, not a spurious 0/0 score.
        """
        empty_period = {k: None for k in [
            "net_income", "operating_cash_flow", "total_assets",
            "gross_profit", "total_revenue", "long_term_debt",
            "current_assets", "current_liabilities", "shares_outstanding",
        ]}
        inputs = {"current": empty_period, "prior": empty_period}

        result = quality_signal(inputs)

        assert result is None
