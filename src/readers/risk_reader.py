"""
src/tools/risk_reader.py

Fetches raw time-series data needed for Risk Signal Engine calculations
(Beta, Sharpe Ratio, VaR, Max Drawdown) — the ticker's own price history
plus the market benchmark (S&P 500) and risk-free rate, none of which
are available from snapshot_reader.py's single point-in-time .info call.

Split out from the former market_data.py on 2026-07-27 (alongside
snapshot_reader.py and consensus_reader.py) — one reader file per data
point, named for the signal it serves, so a future "which data does
this question need" routing node can address each data source
independently.
"""

import yfinance as yf


def get_risk_inputs(ticker: str, period: str = "2y") -> dict | None:
    """
    Fetches raw time-series data needed for Risk Signal Engine calculations
    (Beta, Sharpe Ratio, VaR, Max Drawdown).

    Unlike snapshot_reader.get_stock_snapshot(), this pulls a longer price
    history and includes the market benchmark (S&P 500) and risk-free rate
    — inputs that only the quant risk calculations need, not the point-in-time
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
