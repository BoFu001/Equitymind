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
        "pe_ratio":       float(DEFAULT_PE),  # matches peer avg for AAPL
        "peg_ratio":      1.0,
        "price_to_sales": float(DEFAULT_PS),
        "target_mean":    100.0,  # no upside by default → neutral upside_score
    }
    defaults.update(kwargs)
    return defaults


# ─────────────────────────────────────────────
# Tier 1 — P/E + PEG + analyst upside (most reliable)
# ─────────────────────────────────────────────

class TestTier1:

    def test_fairly_valued(self):
        """P/E equals sector average, PEG = 1, no upside → score near zero."""
        data   = make_market_data()  # pe_ratio = sector avg, target_mean = current_price
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "pe_peg_upside"
        assert result["reference_only"]  is False
        assert -0.2 <= result["valuation_score"] <= 0.2
        assert result["valuation_label"] == "fairly valued"

    def test_undervalued(self):
        """P/E well below sector average and PEG < 1 → undervalued."""
        data   = make_market_data(pe_ratio=5.0, peg_ratio=0.3, target_mean=130.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "pe_peg_upside"
        assert result["valuation_score"] >  0.2
        assert result["valuation_label"] == "undervalued"

    def test_overvalued(self):
        """P/E well above sector average and PEG > 1 → overvalued."""
        data   = make_market_data(pe_ratio=300.0, peg_ratio=3.0, target_mean=90.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "pe_peg_upside"
        assert result["valuation_score"] <  -0.2
        assert result["valuation_label"] == "overvalued"

    def test_detail_string_non_empty(self):
        """Result must include a non-empty plain-English detail string."""
        result = valuation_signal(make_market_data())
        assert result is not None
        assert "detail" in result
        assert len(result["detail"]) > 0

    def test_upside_pct_calculated_correctly(self):
        """Upside percentage should equal (target - price) / price * 100."""
        data   = make_market_data(current_price=100.0, target_mean=120.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["upside_pct"] == 20.0

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


# ─────────────────────────────────────────────
# Tier 2 — P/E + analyst upside (no PEG available)
# ─────────────────────────────────────────────

class TestTier2:

    def test_falls_to_tier2_when_peg_none(self):
        """Missing PEG should trigger Tier 2 fallback."""
        data   = make_market_data(peg_ratio=None)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]         == "pe_upside"
        assert result["reference_only"] is False

    def test_falls_to_tier2_when_peg_negative(self):
        """Negative PEG (e.g. negative earnings growth) should trigger Tier 2."""
        data   = make_market_data(peg_ratio=-1.5)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"] == "pe_upside"

    def test_detail_mentions_peg_not_available(self):
        """Tier 2 detail must note that PEG was not available."""
        data   = make_market_data(peg_ratio=None)
        result = valuation_signal(data)
        assert result is not None
        assert "PEG not available" in result["detail"]

    def test_stale_benchmark_present(self):
        """Tier 2 result must include stale_benchmark field."""
        data   = make_market_data(peg_ratio=None)
        result = valuation_signal(data)
        assert result is not None
        assert "stale_benchmark" in result
        assert isinstance(result["stale_benchmark"], bool)


# ─────────────────────────────────────────────
# Tier 3 — P/S only (loss-making companies, reference only)
# ─────────────────────────────────────────────

class TestTier3:

    def test_loss_making_company_uses_ps(self):
        """Negative P/E with available P/S → Tier 3 (reference only)."""
        data   = make_market_data(pe_ratio=-10.0, peg_ratio=None, price_to_sales=8.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]          == "ps_only"
        assert result["reference_only"]  is True
        assert result["valuation_label"] == "reference only"

    def test_no_pe_falls_to_tier3(self):
        """No P/E data at all → Tier 3 if P/S is available."""
        data   = make_market_data(pe_ratio=None, peg_ratio=None, price_to_sales=5.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["method"]         == "ps_only"
        assert result["reference_only"] is True

    def test_detail_warns_reference_only(self):
        """Tier 3 detail must explicitly warn the score is for reference only."""
        data   = make_market_data(pe_ratio=None, peg_ratio=None, price_to_sales=5.0)
        result = valuation_signal(data)
        assert result is not None
        assert "reference only" in result["detail"].lower()

    def test_pe_vs_peers_is_none(self):
        """Tier 3 must not populate pe_vs_peers (P/E is not used)."""
        data   = make_market_data(pe_ratio=None, peg_ratio=None, price_to_sales=6.0)
        result = valuation_signal(data)
        assert result is not None
        assert result["pe_vs_peers"] is None

    def test_stale_benchmark_present(self):
        """Tier 3 result must include stale_benchmark field."""
        data   = make_market_data(pe_ratio=None, peg_ratio=None, price_to_sales=5.0)
        result = valuation_signal(data)
        assert result is not None
        assert "stale_benchmark" in result
        assert isinstance(result["stale_benchmark"], bool)


# ─────────────────────────────────────────────
# Tier 4 — No usable data → returns None
# ─────────────────────────────────────────────

class TestTier4:

    def test_no_data_returns_none(self):
        """No P/E, PEG, or P/S → must return None to prevent hallucination."""
        data   = make_market_data(pe_ratio=None, peg_ratio=None, price_to_sales=None)
        result = valuation_signal(data)
        assert result is None

    def test_zero_values_return_none(self):
        """Zero P/E and P/S must be treated as missing data."""
        data   = make_market_data(pe_ratio=0, peg_ratio=0, price_to_sales=0)
        result = valuation_signal(data)
        assert result is None


# ─────────────────────────────────────────────
# Score bounds — extreme inputs must never exceed ±1.0
# ─────────────────────────────────────────────

class TestScoreBounds:

    def test_score_clamped_on_extreme_cheap(self):
        """Extremely cheap stock must not produce score > +1.0."""
        data   = make_market_data(
            pe_ratio=0.5, peg_ratio=0.01, target_mean=10000.0, current_price=100.0
        )
        result = valuation_signal(data)
        assert result is not None
        assert result["valuation_score"] <= 1.0

    def test_score_clamped_on_extreme_expensive(self):
        """Extremely expensive stock must not produce score < -1.0."""
        data   = make_market_data(
            pe_ratio=5000.0, peg_ratio=20.0, target_mean=1.0, current_price=100.0
        )
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

    def test_default_ps_is_positive_number(self):
        """DEFAULT_PS must be a positive number loaded from JSON."""
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


# ─────────────────────────────────────────────
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