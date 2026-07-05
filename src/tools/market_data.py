import yfinance as yf
import ta


def get_stock_data(ticker: str) -> dict | None:
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

        # Technical indicators from last 6 months of price data
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