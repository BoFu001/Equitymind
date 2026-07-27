import yfinance as yf
import pandas as pd


def get_stock_snapshot(ticker: str) -> dict | None:
    """
    Fetches fundamentals and technical indicators for a ticker.
    Returns a dict with price, P/E, revenue, RSI, MACD, SMA or None if failed.
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info

        # Basic fundamentals
        market_data = {
            "ticker":             ticker,
            "company_name":       info.get("longName", ticker),
            "current_price":      info.get("currentPrice"),
            "market_cap":         info.get("marketCap"),
            "pe_ratio":           info.get("trailingPE"),
            "forward_pe":         info.get("forwardPE"),
            "revenue":            info.get("totalRevenue"),
            "profit_margin":      info.get("profitMargins"),
            "52w_high":           info.get("fiftyTwoWeekHigh"),
            "52w_low":            info.get("fiftyTwoWeekLow"),
            "sector":             info.get("sector"),
            "industry":           info.get("industry"),
            # Earnings and dividends
            "eps_trailing":       info.get("trailingEps"),
            "eps_forward":        info.get("forwardEps"),
            "dividend_yield":     info.get("dividendYield"),
            # Analyst targets
            "target_high":        info.get("targetHighPrice"),
            "target_low":         info.get("targetLowPrice"),
            "target_mean":        info.get("targetMeanPrice"),
            "recommendation":     info.get("recommendationKey"),
            "analyst_count":       info.get("numberOfAnalystOpinions"),
            "recommendation_mean": info.get("recommendationMean"),
            # Valuation ratios for Layer 2 quant engine
            "price_to_sales":     info.get("priceToSalesTrailing12Months"),
            "price_to_book":      info.get("priceToBook"),
        }

        # NOTE: RSI/MACD/SMA technical indicators were deliberately removed —
        # they lack independent, large-sample academic validation (see
        # momentum_signal.py docstring). Momentum is now measured via
        # 12-1 month momentum and 52-week high position, computed from
        # precomputed universe benchmarks, not from this function.

        return market_data

    except Exception as e:
        print(f"  [get_stock_data] Error fetching {ticker}: {e}")
        return None
    


def get_risk_inputs(ticker: str, period: str = "2y") -> dict | None:
    """
    Fetches raw time-series data needed for Risk Signal Engine calculations
    (Beta, Sharpe Ratio, VaR, Max Drawdown).

    Unlike get_stock_snapshot(), this pulls a longer price history and
    includes the market benchmark (S&P 500) and risk-free rate — inputs
    that only the quant risk calculations need, not the point-in-time
    snapshot used elsewhere.

    Returns None if the stock's own price history is unavailable, since
    none of the risk metrics can be computed without it. The market
    benchmark and risk-free rate are allowed to be missing independently —
    callers (risk_signal.py) decide how to degrade gracefully when that
    happens.

    Args:
        ticker: stock ticker symbol, e.g. "AAPL"
        period: yfinance period string, default "2y" (~504 trading days)

    Returns:
        dict with keys:
            - stock_prices:   pd.Series of daily closing prices for the ticker
            - market_prices:  pd.Series of daily closing prices for ^GSPC (S&P 500),
                               or None if the benchmark fetch failed
            - risk_free_rate: float, latest ^TNX yield as a decimal (e.g. 0.042
                               for 4.2%), or None if the fetch failed
        or None if the stock's own price history could not be fetched.
    """
    try:
        stock_hist = yf.Ticker(ticker).history(period=period)["Close"]

        # Without the stock's own price series, none of the four risk
        # metrics (Beta, Sharpe, VaR, Max Drawdown) can be computed at all.
        if stock_hist.empty:
            return None

        # Market benchmark (^GSPC) — needed only for Beta.
        # Allowed to fail independently; Beta alone gets skipped downstream.
        try:
            market_hist = yf.Ticker("^GSPC").history(period=period)["Close"]
            market_prices = market_hist if not market_hist.empty else None
        except Exception as e:
            print(f"  [get_risk_inputs] Could not fetch ^GSPC for {ticker}: {e}")
            market_prices = None

        # Risk-free rate (^TNX, 10-year Treasury yield) — needed only for Sharpe.
        # yfinance returns ^TNX already scaled as a percentage (e.g. 4.2),
        # so divide by 100 to get a decimal rate (0.042).
        try:
            tnx_hist = yf.Ticker("^TNX").history(period="5d")["Close"]
            risk_free_rate = round(tnx_hist.iloc[-1] / 100, 4) if not tnx_hist.empty else None
        except Exception as e:
            print(f"  [get_risk_inputs] Could not fetch ^TNX for {ticker}: {e}")
            risk_free_rate = None

        return {
            "stock_prices":   stock_hist,
            "market_prices":  market_prices,
            "risk_free_rate": risk_free_rate,
        }

    except Exception as e:
        print(f"  [get_risk_inputs] Error fetching {ticker}: {e}")
        return None

def get_consensus_inputs(ticker: str) -> dict | None:
    """
    Fetches historical monthly analyst recommendation distributions needed
    for Consensus Signal Engine calculations (recommendation trend).

    Unlike get_stock_snapshot(), this pulls a multi-period history of
    analyst rating counts (yfinance's .recommendations, typically covering
    the most recent ~4 months) rather than a single point-in-time mean —
    the trend calculation needs at least 2 periods to compare.

    Note: the analyst count contributing to each period can fluctuate
    slightly month to month (e.g. individual analysts starting/stopping
    coverage) — this is normal and handled by using proportions (weighted
    averages), not raw counts, in the downstream trend calculation.

    Returns None if fewer than 2 periods of history are available, since
    a trend cannot be computed from a single snapshot.

    Args:
        ticker: stock ticker symbol, e.g. "AAPL"

    Returns:
        dict with keys:
            - periods: list of dicts, each with "period" (e.g. "0m", "-1m")
                       and counts for strongBuy/buy/hold/sell/strongSell,
                       ordered most-recent-first (matches yfinance's own order)
        or None if fewer than 2 periods of history are available.
    """
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations

        if recommendations is None or len(recommendations) < 2:
            return None

        periods = []
        for _, row in recommendations.iterrows():
            periods.append({
                "period":     row["period"],
                "strongBuy":  int(row["strongBuy"]),
                "buy":        int(row["buy"]),
                "hold":       int(row["hold"]),
                "sell":       int(row["sell"]),
                "strongSell": int(row["strongSell"]),
            })

        return {"periods": periods}

    except Exception as e:
        print(f"  [get_consensus_inputs] Error fetching {ticker}: {e}")
        return None
