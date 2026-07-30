"""
Unit tests for Layer 2 Quantitative Signal Engines.

All tests use mock market data — no real API calls are made.
Benchmark values are loaded dynamically from
src/quant/data/valuation_benchmarks.json so tests remain valid after
benchmark updates via scripts/update_valuation_benchmarks.py.
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

def make_valuation_inputs(**kwargs) -> dict:
    """
    Return a minimal valuation_inputs dict for testing.
    All fields mirror those returned by get_valuation_inputs().
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
        data   = make_valuation_inputs()
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "pe_pb"
        assert result["reference_only"]  is False
        assert -0.2 <= result["valuation_score"] <= 0.2
        assert result["valuation_label"] == "fairly valued"

    def test_undervalued(self):
        """P/E and P/B well below peer average → undervalued."""
        data   = make_valuation_inputs(pe_ratio=5.0, price_to_book=1.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "pe_pb"
        assert result["valuation_score"] >  0.2
        assert result["valuation_label"] == "undervalued"

    def test_overvalued(self):
        """P/E and P/B well above peer average → overvalued."""
        data   = make_valuation_inputs(pe_ratio=300.0, price_to_book=100.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "pe_pb"
        assert result["valuation_score"] <  -0.2
        assert result["valuation_label"] == "overvalued"

    def test_detail_string_non_empty(self):
        """Result must include a non-empty plain-English detail string."""
        result = valuation_signal(make_valuation_inputs())
        assert result is not None
        assert "detail" in result
        assert len(result["detail"]) > 0

    def test_pe_vs_peers_string_present(self):
        """pe_vs_peers should describe the comparison as a readable string."""
        result = valuation_signal(make_valuation_inputs())
        assert result is not None
        assert result["pe_vs_peers"] is not None
        assert "vs peer avg" in result["pe_vs_peers"]

    def test_stale_benchmark_is_bool(self):
        """stale_benchmark must be a boolean."""
        result = valuation_signal(make_valuation_inputs())
        assert result is not None
        assert isinstance(result["stale_benchmark"], bool)

    def test_missing_pb_degrades_to_pe_only(self):
        """Missing P/B should degrade gracefully to a P/E-only score."""
        data   = make_valuation_inputs(price_to_book=None)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"] == "pe_pb"
        assert "P/B data unavailable" in result["detail"]

    def test_zero_pb_treated_as_missing(self):
        """A P/B of exactly 0 should be treated as missing, not a real ratio."""
        data   = make_valuation_inputs(price_to_book=0)
        result = valuation_signal(data)
        assert result is not None
        assert "P/B data unavailable" in result["detail"]


# ─────────────────────────────────────────────
# Tier 2 — P/S only (loss-making companies, reference only)
# ─────────────────────────────────────────────

class TestTier2:

    def test_loss_making_company_uses_ps(self):
        """Negative P/E with available P/S → Tier 2 (reference only)."""
        data   = make_valuation_inputs(pe_ratio=-10.0, price_to_book=None, price_to_sales=8.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "ps_only"
        assert result["reference_only"]  is True
        assert result["valuation_label"] == "reference only"

    def test_no_pe_falls_to_tier2(self):
        """No P/E data at all → Tier 2 if P/S is available."""
        data   = make_valuation_inputs(pe_ratio=None, price_to_book=None, price_to_sales=5.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]         == "ps_only"
        assert result["reference_only"] is True

    def test_detail_warns_reference_only(self):
        """Tier 2 detail must explicitly warn the score is for reference only."""
        data   = make_valuation_inputs(pe_ratio=None, price_to_book=None, price_to_sales=5.0)
        result = valuation_signal(data)
        assert result is not None
        assert "reference only" in result["detail"].lower()

    def test_pe_vs_peers_is_none(self):
        """Tier 2 must not populate pe_vs_peers (P/E is not used)."""
        data   = make_valuation_inputs(pe_ratio=None, price_to_book=None, price_to_sales=6.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["pe_vs_peers"] is None

    def test_stale_benchmark_present(self):
        """Tier 2 result must include stale_benchmark field."""
        data   = make_valuation_inputs(pe_ratio=None, price_to_book=None, price_to_sales=5.0)
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
        data   = make_valuation_inputs(pe_ratio=None, price_to_book=None, price_to_sales=None)
        result = valuation_signal(data)
        assert result is None

    def test_zero_values_return_none(self):
        """Zero P/E and P/S must be treated as missing data."""
        data   = make_valuation_inputs(pe_ratio=0, price_to_book=0, price_to_sales=0)
        result = valuation_signal(data)
        assert result is None


# ─────────────────────────────────────────────
# Score bounds — extreme inputs must never exceed ±1.0
# ─────────────────────────────────────────────

class TestScoreBounds:

    def test_score_clamped_on_extreme_cheap(self):
        """Extremely cheap stock must not produce score > +1.0."""
        data   = make_valuation_inputs(pe_ratio=0.5, price_to_book=0.1)
        result = valuation_signal(data)
        assert result is not None
        assert result["valuation_score"] <= 1.0

    def test_score_clamped_on_extreme_expensive(self):
        """Extremely expensive stock must not produce score < -1.0."""
        data   = make_valuation_inputs(pe_ratio=5000.0, price_to_book=1000.0)
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
        data   = make_valuation_inputs(ticker="UNKNOWN")
        result = valuation_signal(data)
        assert result is not None
        assert str(int(DEFAULT_PE)) in result["pe_vs_peers"]

    def test_benchmarks_stale_is_bool(self):
        """BENCHMARKS_STALE must be a boolean."""
        assert isinstance(BENCHMARKS_STALE, bool)


# Helpers — mock momentum benchmark entries
# ─────────────────────────────────────────────

import src.quant.momentum_signal as momentum_module
from src.quant.momentum_signal import momentum_signal


def make_momentum_entry(**kwargs) -> dict:
    """Create a minimal mock momentum_benchmarks.json entry for testing."""
    defaults = {
        "momentum_12_1_pct":        10.0,
        "momentum_12_1_percentile": 0.5,
        "position_52w":            0.5,
        "position_52w_percentile":  0.5,
    }
    defaults.update(kwargs)
    return defaults


def set_mock_benchmarks(monkeypatch, entries: dict):
    """Replace the module-level benchmark lookup with mock data for a test."""
    monkeypatch.setattr(momentum_module, "_BENCHMARKS_DATA", entries)


# ─────────────────────────────────────────────
# Momentum Signal — two independent sub-signals, not combined
# ─────────────────────────────────────────────

class TestMomentumNeutral:

    def test_median_percentile_is_neutral(self, monkeypatch):
        """Percentile of 0.5 (median of universe) -> neutral for both sub-signals."""
        set_mock_benchmarks(monkeypatch, {"TEST": make_momentum_entry()})
        result = momentum_signal("TEST")

        assert result is not None
        assert result["momentum_12_1_label"] == "neutral"
        assert result["position_52w_label"] == "neutral"

    def test_returns_all_fields(self, monkeypatch):
        """Result must include all required fields for both sub-signals."""
        set_mock_benchmarks(monkeypatch, {"TEST": make_momentum_entry()})
        result = momentum_signal("TEST")

        assert result is not None
        for field in [
            "momentum_12_1_pct", "momentum_12_1_percentile", "momentum_12_1_score", "momentum_12_1_label",
            "position_52w", "position_52w_percentile", "position_52w_score", "position_52w_label",
            "stale_benchmark", "detail",
        ]:
            assert field in result

    def test_no_composite_score_exists(self, monkeypatch):
        """
        There should be no combined momentum_score/momentum_label —
        12-1 momentum and 52-week position answer different questions
        (cumulative return vs current position) and must remain
        independent, not averaged into one number.
        """
        set_mock_benchmarks(monkeypatch, {"TEST": make_momentum_entry()})
        result = momentum_signal("TEST")

        assert result is not None
        assert "momentum_score" not in result
        assert "momentum_label" not in result

    def test_top_percentile_is_strong(self, monkeypatch):
        """Percentile of 1.0 (top of universe) -> strong for both sub-signals."""
        set_mock_benchmarks(monkeypatch, {"TEST": make_momentum_entry(
            momentum_12_1_percentile=1.0, position_52w_percentile=1.0
        )})
        result = momentum_signal("TEST")

        assert result is not None
        assert result["momentum_12_1_label"] == "strong"
        assert result["position_52w_label"] == "strong"

    def test_bottom_percentile_is_weak(self, monkeypatch):
        """Percentile of 0.0 (bottom of universe) -> weak for both sub-signals."""
        set_mock_benchmarks(monkeypatch, {"TEST": make_momentum_entry(
            momentum_12_1_percentile=0.0, position_52w_percentile=0.0
        )})
        result = momentum_signal("TEST")

        assert result is not None
        assert result["momentum_12_1_label"] == "weak"
        assert result["position_52w_label"] == "weak"

    def test_scores_in_range(self, monkeypatch):
        """Both sub-signal scores must fall within [-1.0, +1.0]."""
        set_mock_benchmarks(monkeypatch, {"TEST": make_momentum_entry(
            momentum_12_1_percentile=1.0, position_52w_percentile=0.0
        )})
        result = momentum_signal("TEST")

        assert result is not None
        assert -1.0 <= result["momentum_12_1_score"] <= 1.0
        assert -1.0 <= result["position_52w_score"] <= 1.0

    def test_ticker_not_in_universe_returns_none(self, monkeypatch):
        """A ticker missing from the batch-computed universe -> None entirely."""
        set_mock_benchmarks(monkeypatch, {})
        result = momentum_signal("UNKNOWN_TICKER")
        assert result is None

    def test_detail_includes_limitation_disclosure(self, monkeypatch):
        """Every result must disclose the group-level / not-a-prediction limitation."""
        set_mock_benchmarks(monkeypatch, {"TEST": make_momentum_entry()})
        result = momentum_signal("TEST")

        assert result is not None
        assert "not a prediction" in result["detail"]

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
        risk_inputs = make_risk_inputs(n_days=504, beta_true=1.5)

        result = risk_signal(risk_inputs)

        assert result is not None
        assert result["beta"] is not None
        assert result["beta"]["raw_beta"] == pytest.approx(1.5, abs=0.01)

        expected_adjusted = BLUME_RAW_WEIGHT * 1.5 + BLUME_MARKET_WEIGHT * 1.0
        assert result["beta"]["adjusted_beta"] == pytest.approx(expected_adjusted, abs=0.01)

    def test_each_sub_score_in_range(self):
        """Each independent sub-signal score must fall within [-1.0, +1.0]."""
        risk_inputs = make_risk_inputs(n_days=504, beta_true=1.0)

        result = risk_signal(risk_inputs)

        assert result is not None
        assert -1.0 <= result["beta"]["beta_score"] <= 1.0
        assert -1.0 <= result["sharpe"]["sharpe_score"] <= 1.0
        assert -1.0 <= result["var"]["var_score"] <= 1.0
        assert -1.0 <= result["max_drawdown"]["drawdown_score"] <= 1.0

    def test_no_composite_score_exists(self):
        """
        There should be no combined risk_score/risk_level — Beta, Sharpe,
        VaR, and Max Drawdown each answer a different risk question and
        must remain independent, not averaged into one number.
        """
        risk_inputs = make_risk_inputs(n_days=504, beta_true=1.0)

        result = risk_signal(risk_inputs)

        assert result is not None
        assert "risk_score" not in result
        assert "risk_level" not in result

    def test_low_confidence_false_with_full_window(self):
        """A full 504-day (2-year) window should not be flagged low_confidence."""
        risk_inputs = make_risk_inputs(n_days=504)

        result = risk_signal(risk_inputs)

        assert result["low_confidence"] is False


class TestRiskSignalMissingMarketData:
    def test_beta_skipped_when_market_prices_missing(self):
        """
        ^GSPC unavailable -> Beta should be skipped (None), but Sharpe,
        VaR, and Max Drawdown should still compute normally, and the
        composite score should re-normalise over the remaining three.
        """
        risk_inputs = make_risk_inputs(n_days=504, include_market=False)

        result = risk_signal(risk_inputs)

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
        risk_inputs = make_risk_inputs(n_days=504, risk_free_rate=None)

        result = risk_signal(risk_inputs)

        assert result is not None
        assert result["sharpe"] is not None


class TestRiskSignalInsufficientData:
    def test_returns_none_when_risk_inputs_is_none(self):
        """get_risk_inputs() itself failed (e.g. network error) -> None."""
        result = risk_signal(None)
        assert result is None

    def test_returns_none_below_min_trading_days(self):
        """Fewer than MIN_TRADING_DAYS (60) -> too noisy to report, return None."""
        risk_inputs = make_risk_inputs(n_days=MIN_TRADING_DAYS - 1)

        result = risk_signal(risk_inputs)

        assert result is None

    def test_low_confidence_true_between_thresholds(self):
        """
        Between MIN_TRADING_DAYS (60) and LOW_CONFIDENCE_THRESHOLD (252):
        computable, but flagged as low_confidence.
        """
        n_days = (MIN_TRADING_DAYS + LOW_CONFIDENCE_THRESHOLD) // 2
        risk_inputs = make_risk_inputs(n_days=n_days)

        result = risk_signal(risk_inputs)

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


# ─────────────────────────────────────────────
# Consensus Signal Engine (Analyst Sentiment)
# ─────────────────────────────────────────────
from src.quant.consensus_signal import consensus_signal
from src.quant.consensus_signal_config import (
    MIN_ANALYST_COUNT,
    WIDE_TARGET_RANGE_RATIO,
)


def make_consensus_snapshot(**kwargs) -> dict:
    """Minimal mock consensus snapshot dict for consensus signal testing."""
    defaults = {
        "recommendation_mean": 3.0,   # neutral "hold"
        "target_mean":         100.0,
        "current_price":       100.0,  # no upside by default
        "target_high":         110.0,
        "target_low":          90.0,
        "analyst_count":       20,
    }
    defaults.update(kwargs)
    return defaults


def make_consensus_inputs(**overrides) -> dict:
    """
    Mock consensus_inputs with 2 periods by default: current ("0m") and
    oldest ("-3m"), both identical (stable/no trend) unless overridden.
    """
    default_period = {
        "period": "0m", "strongBuy": 5, "buy": 10, "hold": 5, "sell": 0, "strongSell": 0
    }
    current = {**default_period, **overrides.get("current", {})}
    oldest  = {**default_period, "period": "-3m", **overrides.get("oldest", {})}
    return {"periods": [current, oldest]}


def make_consensus_data(snapshot: dict | None = None, trend: dict | None = None) -> dict:
    """
    Wraps make_consensus_snapshot() and make_consensus_inputs() into
    the {"snapshot": ..., "trend": ...} shape consensus_signal() expects
    as of 2026-07-27 (single-argument signature — see consensus_signal.py
    for why snapshot and trend are now fetched independently rather than
    passed as two separate arguments).

    snapshot=None uses make_consensus_snapshot()'s defaults; pass an
    explicit dict (including {}) to override. trend=None means no rating
    history available (matches consensus_signal()'s "trend independently
    unavailable" degradation path) — pass make_consensus_inputs(...) 
    explicitly when a test needs trend data.
    """
    return {
        "snapshot": snapshot if snapshot is not None else make_consensus_snapshot(),
        "trend": trend,
    }


class TestConsensusNormalCase:
    def test_neutral_case(self):
        """Hold recommendation, no upside, stable trend -> each sub-signal neutral."""
        snapshot = make_consensus_snapshot()
        consensus_inputs = make_consensus_inputs()
        result = consensus_signal(make_consensus_data(snapshot, consensus_inputs))

        assert result is not None
        assert result["recommendation_label"] == "neutral"
        assert result["upside_label"] == "neutral"
        assert result["trend_label"] == "stable"

    def test_bullish_case(self):
        """Strong buy, large upside, improving trend -> each sub-signal bullish/improving."""
        snapshot = make_consensus_snapshot(
            recommendation_mean=1.2, target_mean=150.0, current_price=100.0
        )
        consensus_inputs = make_consensus_inputs(
            current={"strongBuy": 15, "buy": 5, "hold": 0, "sell": 0, "strongSell": 0},
            oldest={"strongBuy": 5, "buy": 10, "hold": 5, "sell": 0, "strongSell": 0},
        )
        result = consensus_signal(make_consensus_data(snapshot, consensus_inputs))

        assert result is not None
        assert result["recommendation_label"] == "bullish"
        assert result["upside_label"] == "bullish"
        assert result["trend_label"] == "improving"

    def test_bearish_case(self):
        """Strong sell, negative upside, deteriorating trend -> each sub-signal bearish/deteriorating."""
        snapshot = make_consensus_snapshot(
            recommendation_mean=4.5, target_mean=70.0, current_price=100.0
        )
        consensus_inputs = make_consensus_inputs(
            current={"strongBuy": 0, "buy": 0, "hold": 5, "sell": 10, "strongSell": 5},
            oldest={"strongBuy": 5, "buy": 10, "hold": 5, "sell": 0, "strongSell": 0},
        )
        result = consensus_signal(make_consensus_data(snapshot, consensus_inputs))

        assert result is not None
        assert result["recommendation_label"] == "bearish"
        assert result["upside_label"] == "bearish"
        assert result["trend_label"] == "deteriorating"

    def test_each_sub_score_in_range(self):
        """Each independent sub-signal score must fall within [-1.0, +1.0]."""
        snapshot = make_consensus_snapshot(
            recommendation_mean=1.0, target_mean=1000.0, current_price=100.0
        )
        consensus_inputs = make_consensus_inputs()
        result = consensus_signal(make_consensus_data(snapshot, consensus_inputs))

        assert -1.0 <= result["recommendation_score"] <= 1.0
        assert -1.0 <= result["upside_score"] <= 1.0
        if result["trend_score"] is not None:
            assert -1.0 <= result["trend_score"] <= 1.0

    def test_no_composite_score_exists(self):
        """
        There should be no combined consensus_score/consensus_label —
        recommendation, upside, and trend answer different questions
        (current standing / future price target / recent directional
        change) and must remain independent, not averaged into one number.
        """
        result = consensus_signal(make_consensus_data(make_consensus_snapshot(), make_consensus_inputs()))
        assert result is not None
        assert "consensus_score" not in result
        assert "consensus_label" not in result

    def test_detail_string_non_empty(self):
        result = consensus_signal(make_consensus_data(make_consensus_snapshot(), make_consensus_inputs()))
        assert result is not None
        assert len(result["detail"]) > 0

    def test_detail_includes_bias_disclosure(self):
        """Every result must disclose the systematic optimism bias in analyst ratings."""
        result = consensus_signal(make_consensus_data(make_consensus_snapshot(), make_consensus_inputs()))
        assert result is not None
        assert "optimism bias" in result["detail"]


class TestConsensusTrendCalculation:
    def test_improving_trend_is_positive(self):
        """Weighted rating moving toward 'buy' (lower number) -> positive trend_score."""
        consensus_inputs = make_consensus_inputs(
            current={"strongBuy": 15, "buy": 5, "hold": 0, "sell": 0, "strongSell": 0},
            oldest={"strongBuy": 0, "buy": 5, "hold": 15, "sell": 0, "strongSell": 0},
        )
        result = consensus_signal(make_consensus_data(make_consensus_snapshot(), consensus_inputs))

        assert result["trend_score"] > 0

    def test_deteriorating_trend_is_negative(self):
        """Weighted rating moving toward 'sell' (higher number) -> negative trend_score."""
        consensus_inputs = make_consensus_inputs(
            current={"strongBuy": 0, "buy": 5, "hold": 15, "sell": 0, "strongSell": 0},
            oldest={"strongBuy": 15, "buy": 5, "hold": 0, "sell": 0, "strongSell": 0},
        )
        result = consensus_signal(make_consensus_data(make_consensus_snapshot(), consensus_inputs))

        assert result["trend_score"] < 0

    def test_fluctuating_analyst_count_does_not_break_trend(self):
        """
        Total analyst count differing slightly between periods (e.g. 20 vs
        21, as seen in real AAPL/JPM/KO data) should not cause errors —
        proportions handle this naturally.
        """
        consensus_inputs = make_consensus_inputs(
            current={"strongBuy": 6, "buy": 22, "hold": 16, "sell": 1, "strongSell": 2},  # total 47
            oldest={"strongBuy": 7, "buy": 25, "hold": 14, "sell": 1, "strongSell": 1},   # total 48
        )
        result = consensus_signal(make_consensus_data(make_consensus_snapshot(), consensus_inputs))

        assert result is not None
        assert result["trend_score"] is not None


class TestConsensusMissingData:
    def test_missing_consensus_inputs_trend_is_none(self):
        """
        If consensus_inputs is None (e.g. no rating history available),
        trend_score and trend_label should both be None — recommendation
        and upside are independent and unaffected, since there is no
        composite score to reweight (each sub-signal stands on its own).
        """
        result = consensus_signal(make_consensus_data(make_consensus_snapshot(), None))

        assert result is not None
        assert result["trend_score"] is None
        assert result["trend_label"] is None
        # recommendation and upside are computed independently of trend
        assert result["recommendation_score"] is not None
        assert result["recommendation_label"] is not None
        assert result["upside_score"] is not None
        assert result["upside_label"] is not None

    def test_single_period_history_returns_none_trend(self):
        """Only 1 period of rating history -> trend cannot be computed."""
        consensus_inputs = {"periods": [
            {"period": "0m", "strongBuy": 5, "buy": 10, "hold": 5, "sell": 0, "strongSell": 0}
        ]}
        result = consensus_signal(make_consensus_data(make_consensus_snapshot(), consensus_inputs))

        assert result is not None
        assert result["trend_score"] is None

    def test_missing_recommendation_mean_returns_none(self):
        """No recommendation_mean at all -> entire signal returns None."""
        snapshot = make_consensus_snapshot(recommendation_mean=None)
        result = consensus_signal(make_consensus_data(snapshot, make_consensus_inputs()))
        assert result is None

    def test_missing_target_mean_returns_none(self):
        """No target_mean at all -> entire signal returns None."""
        snapshot = make_consensus_snapshot(target_mean=None)
        result = consensus_signal(make_consensus_data(snapshot, make_consensus_inputs()))
        assert result is None


class TestConsensusDisclosures:
    def test_low_confidence_flagged_when_analyst_count_below_threshold(self):
        snapshot = make_consensus_snapshot(analyst_count=MIN_ANALYST_COUNT - 1)
        result = consensus_signal(make_consensus_data(snapshot, make_consensus_inputs()))

        assert result["low_confidence"] is True

    def test_low_confidence_not_flagged_when_analyst_count_sufficient(self):
        snapshot = make_consensus_snapshot(analyst_count=MIN_ANALYST_COUNT + 10)
        result = consensus_signal(make_consensus_data(snapshot, make_consensus_inputs()))

        assert result["low_confidence"] is False

    def test_wide_dispersion_flagged(self):
        """target_high > 3x target_low should trigger the dispersion warning."""
        snapshot = make_consensus_snapshot(target_high=400.0, target_low=100.0)
        result = consensus_signal(make_consensus_data(snapshot, make_consensus_inputs()))

        assert result["wide_dispersion"] is True

    def test_narrow_dispersion_not_flagged(self):
        snapshot = make_consensus_snapshot(target_high=110.0, target_low=90.0)
        result = consensus_signal(make_consensus_data(snapshot, make_consensus_inputs()))

        assert result["wide_dispersion"] is False


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# News Sentiment Signal Engine (Media Tone)
# ─────────────────────────────────────────────
from src.quant.news_sentiment_signal import (
    news_sentiment_signal,
    _aggregate_sentiment,
    _sentiment_label,
    MIN_ARTICLE_COUNT,
)


def make_article(**kwargs) -> dict:
    """
    Create a minimal mock article dict for news sentiment testing.
    NOTE: as of the data-layer/compute-layer split, articles reaching
    news_sentiment_signal() are already filtered (see src/tools/news_reader.py)
    and not yet scored — "sentiment"/"score" here represent what FinBERT
    would assign, used only to construct pre-scored fixtures for these
    aggregation-focused tests (the real function scores title+summary
    itself; these tests bypass that to test aggregation logic in isolation).
    """
    defaults = {
        "title": "Apple (AAPL) Reports Strong Quarterly Earnings",
        "url": "https://example.com/article",
        "published": "2026-07-01T00:00:00.000Z",
        "summary": "Apple beat expectations.",
        "sentiment": "positive",
        "score": 0.9,
    }
    defaults.update(kwargs)
    return defaults


class TestAggregateSentiment:
    """
    Tests _aggregate_sentiment and _sentiment_label directly — these are
    pure functions that trust pre-set sentiment/score fields, unlike
    news_sentiment_signal itself, which calls real FinBERT and overwrites
    those fields with its own live judgment. Testing aggregation here
    keeps these tests fast and fully controllable.
    """

    def test_all_positive_articles(self):
        """All positive articles -> positive label, positive net score."""
        articles = [make_article(sentiment="positive", score=0.9) for _ in range(15)]
        stats = _aggregate_sentiment(articles)
        label = _sentiment_label(stats["net_score"])

        assert label == "positive"
        assert stats["net_score"] > 0
        assert stats["positive_count"] == 15
        assert stats["negative_count"] == 0
        assert stats["total_articles"] == 15

    def test_all_negative_articles(self):
        """All negative articles -> negative label, negative net score."""
        articles = [make_article(sentiment="negative", score=0.9) for _ in range(15)]
        stats = _aggregate_sentiment(articles)
        label = _sentiment_label(stats["net_score"])

        assert label == "negative"
        assert stats["net_score"] < 0

    def test_mixed_sentiment_articles(self):
        """A realistic mix of positive/negative/neutral articles."""
        articles = (
            [make_article(sentiment="positive", score=0.8) for _ in range(6)] +
            [make_article(sentiment="negative", score=0.8) for _ in range(4)] +
            [make_article(sentiment="neutral", score=0.7) for _ in range(3)]
        )
        stats = _aggregate_sentiment(articles)

        assert stats["positive_count"] == 6
        assert stats["negative_count"] == 4
        assert stats["neutral_count"] == 3
        assert stats["total_articles"] == 13

    def test_score_near_zero_is_neutral(self):
        """A roughly balanced mix should land in the neutral buffer zone."""
        articles = (
            [make_article(sentiment="positive", score=0.5) for _ in range(6)] +
            [make_article(sentiment="negative", score=0.5) for _ in range(6)]
        )
        stats = _aggregate_sentiment(articles)
        label = _sentiment_label(stats["net_score"])
        assert label == "neutral"


class TestNewsSentimentSignalEndToEnd:
    """
    Tests news_sentiment_signal as a whole, including its real call to
    FinBERT. Uses unambiguous, clearly-worded headlines so the test
    doesn't depend on FinBERT's judgment of an artificial/generic title
    (e.g. "Apple News 0" carries no real sentiment) — these headlines
    are written to have an unambiguous sentiment a model should get right.
    """

    def test_clearly_positive_headlines_yield_positive_label(self):
        articles = [
            {"title": "Company posts record profit, stock surges to all-time high",
             "url": "https://example.com/a", "published": "2026-07-01T00:00:00.000Z",
             "summary": "Strong earnings beat expectations across all segments."}
            for _ in range(12)
        ]
        result = news_sentiment_signal("AAPL", articles)

        assert result is not None
        assert result["sentiment_label"] == "positive"
        assert result["total_articles"] == 12

    def test_returns_per_article_detail(self):
        """
        Result includes an "articles" list with each article's title,
        url, sentiment, and score — needed by callers (e.g. generate_report)
        that display a source list with links, not just the aggregate score.
        """
        articles = [
            {"title": "Company posts record profit, stock surges to all-time high",
             "url": "https://example.com/a", "published": "2026-07-01T00:00:00.000Z",
             "summary": "Strong earnings beat expectations."}
            for _ in range(12)
        ]
        result = news_sentiment_signal("AAPL", articles)

        assert "articles" in result
        assert len(result["articles"]) == 12
        assert result["articles"][0]["title"] == "Company posts record profit, stock surges to all-time high"
        assert result["articles"][0]["url"] == "https://example.com/a"
        assert "sentiment" in result["articles"][0]
        assert "score" in result["articles"][0]



class TestNewsSentimentLowConfidence:
    def test_below_min_article_count_flags_low_confidence(self):
        """Fewer than MIN_ARTICLE_COUNT relevant articles -> low_confidence True."""
        articles = [make_article(title=f"Apple (AAPL) News {i}", sentiment="positive", score=0.9)
                    for i in range(MIN_ARTICLE_COUNT - 1)]
        result = news_sentiment_signal("AAPL", articles)

        assert result is not None
        assert result["low_confidence"] is True
        assert "small sample size" in result["detail"]

    def test_at_or_above_min_article_count_not_low_confidence(self):
        """MIN_ARTICLE_COUNT or more relevant articles -> low_confidence False."""
        articles = [make_article(title=f"Apple (AAPL) News {i}", sentiment="positive", score=0.9)
                    for i in range(MIN_ARTICLE_COUNT)]
        result = news_sentiment_signal("AAPL", articles)

        assert result is not None
        assert result["low_confidence"] is False


class TestNewsSentimentMissingData:
    def test_empty_article_list_returns_none(self):
        """No articles at all -> None."""
        result = news_sentiment_signal("AAPL", [])
        assert result is None
