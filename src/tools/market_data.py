import yfinance as yf
import ta


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
            "dividend_rate":      info.get("dividendRate"),
            "payout_ratio":       info.get("payoutRatio"),
            # Analyst targets
            "target_high":        info.get("targetHighPrice"),
            "target_low":         info.get("targetLowPrice"),
            "target_mean":        info.get("targetMeanPrice"),
            "recommendation":     info.get("recommendationKey"),
            "analyst_count":       info.get("numberOfAnalystOpinions"),
            "recommendation_mean": info.get("recommendationMean"),
            # Valuation ratios for Layer 2 quant engine
            "peg_ratio":          info.get("trailingPegRatio"),
            "earnings_growth":    info.get("earningsGrowth"),
            "price_to_sales":     info.get("priceToSalesTrailing12Months"),
            "price_to_book":      info.get("priceToBook"),
        }

        # Technical indicators from last 1 year of price data
        hist = stock.history(period="1y")
        if not hist.empty:
            hist["rsi"]         = ta.momentum.RSIIndicator(hist["Close"]).rsi()
            macd                = ta.trend.MACD(hist["Close"])
            hist["macd"]        = macd.macd()
            hist["macd_signal"] = macd.macd_signal()

            market_data["rsi"]         = round(hist["rsi"].iloc[-1], 2)
            market_data["macd"]        = round(hist["macd"].iloc[-1], 4)
            market_data["macd_signal"] = round(hist["macd_signal"].iloc[-1], 4)
            market_data["sma_50"]      = round(hist["Close"].rolling(50).mean().iloc[-1], 2)
            market_data["sma_200"]     = round(hist["Close"].rolling(200).mean().iloc[-1], 2) if len(hist) >= 200 else None

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