"""
src/quant/momentum_signal.py

Momentum Signal Engine — Layer 2 Quantitative Intelligence.

Assesses the current price trend strength and direction using three sub-signals:
    1. RSI Score        — overbought/oversold conditions (weight: 30%)
    2. MACD Score       — trend direction and acceleration (weight: 35%)
    3. Price Position   — price relative to SMA50, SMA200, 52w range (weight: 35%)

Score range: -1.0 (strongly bearish) to +1.0 (strongly bullish)
Label:       "bullish" / "neutral" / "bearish"

Academic references:
    - RSI: Wilder (1978), New Concepts in Technical Trading Systems
    - MACD: Appel (1979), The Moving Average Convergence-Divergence Method
    - 52-week high momentum: George & Hwang (2004), The 52-Week High and Momentum Investing
"""

import math


def momentum_signal(market_data: dict) -> dict | None:
    """
    Compute a momentum signal from yfinance market data.

    Uses a three-tier degradation strategy based on data availability.
    Returns None if no meaningful momentum can be computed.

    Args:
        market_data: dict returned by get_market_data_tool, expected fields:
            - rsi:           float | None
            - macd:          float | None
            - macd_signal:   float | None
            - current_price: float | None
            - sma_50:        float | None
            - sma_200:       float | None  (may be None for recently listed stocks)
            - 52w_high:      float | None
            - 52w_low:       float | None

    Returns:
        dict with keys:
            - momentum_score:  float (-1.0 to +1.0)
            - momentum_label:  str — "bullish" / "neutral" / "bearish"
            - rsi_score:       float | None
            - macd_score:      float | None
            - price_score:     float | None
            - detail:          str — plain English explanation
        or None if insufficient data
    """

    rsi          = market_data.get("rsi")
    macd         = market_data.get("macd")
    macd_signal  = market_data.get("macd_signal")
    price        = market_data.get("current_price")
    sma_50       = market_data.get("sma_50")
    sma_200      = market_data.get("sma_200")
    high_52w     = market_data.get("52w_high")
    low_52w      = market_data.get("52w_low")

    # ── Guard: need at least one signal to compute anything ──────────────────
    if not any([rsi, macd, price]):
        return None

    detail_parts = []

    # ────────────────────────────────────────────────────────────────────────
    # SUB-SIGNAL 1: RSI Score (weight: 30%)
    # Formula: (50 - RSI) / 50
    # RSI=30 → +0.4 (oversold), RSI=70 → -0.4 (overbought), RSI=50 → 0
    # ────────────────────────────────────────────────────────────────────────
    if rsi is not None:
        rsi_score = max(-1.0, min(1.0, (50 - rsi) / 50))
        if rsi >= 70:
            rsi_label = "overbought"
        elif rsi <= 30:
            rsi_label = "oversold"
        else:
            rsi_label = "neutral"
        detail_parts.append(
            f"RSI {round(rsi, 1)} ({rsi_label}, rsi_score={round(rsi_score, 2)})"
        )
    else:
        rsi_score = None

    # ────────────────────────────────────────────────────────────────────────
    # SUB-SIGNAL 2: MACD Score (weight: 35%)
    # Uses tanh normalisation to handle different price magnitudes
    # normalizer = price * 1% so signals are scale-invariant
    # ────────────────────────────────────────────────────────────────────────
    if macd is not None and macd_signal is not None and price:
        diff       = macd - macd_signal
        normalizer = abs(price) * 0.01 if price else 1.0
        macd_score = math.tanh(diff / normalizer) if normalizer != 0 else 0.0
        macd_direction = "bullish" if diff > 0 else "bearish"
        detail_parts.append(
            f"MACD diff {round(diff, 4)} ({macd_direction}, macd_score={round(macd_score, 2)})"
        )
    else:
        macd_score = None

    # ────────────────────────────────────────────────────────────────────────
    # SUB-SIGNAL 3: Price Position Score (weight: 35%)
    # Three components:
    #   a) Price vs SMA50  (30%) — short-term trend
    #   b) Price vs SMA200 (30%) — long-term trend
    #   c) 52-week position (40%) — annual momentum (George & Hwang 2004)
    # Degrades gracefully when SMA200 or 52w data is unavailable
    # ────────────────────────────────────────────────────────────────────────
    if price and sma_50:
        # a) vs SMA50
        sma50_score = max(-1.0, min(1.0, (price - sma_50) / sma_50))

        # b) vs SMA200 (may be None for recently listed stocks)
        if sma_200:
            sma200_score = max(-1.0, min(1.0, (price - sma_200) / sma_200))
        else:
            sma200_score = None

        # c) 52-week position
        if high_52w and low_52w and high_52w != low_52w:
            range_52w      = high_52w - low_52w
            position_52w   = (price - low_52w) / range_52w      # 0=at year low, 1=at year high
            position_score = max(-1.0, min(1.0, position_52w * 2 - 1))  # scale to [-1, 1]
        else:
            position_score = None

        # Weighted combination — adjust weights if components are missing
        if sma200_score is not None and position_score is not None:
            price_score = (
                0.30 * sma50_score +
                0.30 * sma200_score +
                0.40 * position_score
            )
        elif sma200_score is None and position_score is not None:
            # No SMA200 — recently listed stock
            price_score = 0.50 * sma50_score + 0.50 * position_score
        elif sma200_score is not None and position_score is None:
            price_score = 0.50 * sma50_score + 0.50 * sma200_score
        else:
            price_score = sma50_score

        price_score = max(-1.0, min(1.0, price_score))

        detail_parts.append(
            f"Price ${round(price, 2)} vs SMA50 ${sma_50} "
            f"(sma50_score={round(sma50_score, 2)})"
            + (f", vs SMA200 ${sma_200} (sma200_score={round(sma200_score, 2)})" if sma200_score is not None else ", SMA200 not available")
            + (f", 52w position={round(position_52w, 2)} (position_score={round(position_score, 2)})" if position_score is not None else "")
        )
    else:
        price_score = None

    # ────────────────────────────────────────────────────────────────────────
    # COMPOSITE MOMENTUM SCORE
    # Weights adjust dynamically based on available sub-signals
    # ────────────────────────────────────────────────────────────────────────
    available = {
        "rsi":   (rsi_score,   0.30),
        "macd":  (macd_score,  0.35),
        "price": (price_score, 0.35),
    }

    total_weight = sum(w for _, (s, w) in available.items() if s is not None)

    if total_weight == 0:
        return None

    momentum_score = sum(
        s * w for _, (s, w) in available.items() if s is not None
    ) / total_weight  # normalise by actual weight used

    momentum_score = round(max(-1.0, min(1.0, momentum_score)), 4)

    # ── Label ────────────────────────────────────────────────────────────────
    if momentum_score > 0.2:
        momentum_label = "bullish"
    elif momentum_score < -0.2:
        momentum_label = "bearish"
    else:
        momentum_label = "neutral"

    return {
        "momentum_score": momentum_score,
        "momentum_label": momentum_label,
        "rsi_score":      round(rsi_score, 4)   if rsi_score   is not None else None,
        "macd_score":     round(macd_score, 4)  if macd_score  is not None else None,
        "price_score":    round(price_score, 4) if price_score is not None else None,
        "detail":         " | ".join(detail_parts) if detail_parts else "Insufficient data.",
    }