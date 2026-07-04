"""
scripts/update_benchmarks.py

Quarterly benchmark updater for EquityMind Layer 2 Quant Engine.

Fetches real-time sector median metrics from yfinance using a representative
sample of S&P 500 stocks per sector, then writes the results to
src/quant/data/sector_benchmarks.json.

Usage:
    python scripts/update_benchmarks.py

Run quarterly to keep benchmarks current.
Requires: yfinance, numpy, pandas
"""

import json
import numpy as np
from datetime import date
from pathlib import Path

import yfinance as yf

# ─────────────────────────────────────────────
# Representative S&P 500 tickers per sector
# Update this list annually as index composition changes
# Source: S&P 500 sector breakdown (GICS classification)
# ─────────────────────────────────────────────
SECTOR_TICKERS = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ACN",
        "AMD", "TXN", "QCOM", "INTU", "IBM", "NOW", "AMAT", "ADI",
    ],
    "Healthcare": [
        "JNJ", "UNH", "LLY", "ABT", "TMO", "MRK", "DHR", "ISRG",
        "SYK", "BSX", "ELV", "CVS", "CI", "MDT", "AMGN",
    ],
    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX",
        "TJX", "BKNG", "CMG", "F", "GM", "MAR", "HLT",
    ],
    "Consumer Defensive": [
        "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "CL",
        "MDLZ", "GIS", "KHC", "SYY", "KMB", "HSY",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX",
        "VLO", "PXD", "OXY", "HES", "DVN", "FANG",
    ],
    "Financial Services": [
        "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "BLK",
        "SCHW", "AXP", "USB", "PNC", "TFC", "COF", "MMC",
    ],
    "Industrials": [
        "RTX", "HON", "UPS", "CAT", "BA", "DE", "LMT",
        "GE", "MMM", "ETN", "EMR", "ITW", "PH", "FDX",
    ],
    "Basic Materials": [
        "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE",
        "VMC", "MLM", "CF", "MOS", "ALB",
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "SPG", "O", "DLR", "PSA",
        "EXR", "AVB", "EQR", "VTR", "WELL",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL",
        "SRE", "WEC", "ES", "ETR", "PPL",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ",
        "TMUS", "CHTR", "ATVI", "EA", "WBD",
    ],
}

OUTPUT_PATH = Path(__file__).parent.parent / "src" / "quant" / "data" / "sector_benchmarks.json"


def fetch_sector_metrics(sector: str, tickers: list[str]) -> dict:
    """
    Fetch key valuation and risk metrics for a list of tickers,
    then compute the median for each metric across the sector.

    Returns a dict of median metrics for the sector.
    """
    pe_list   = []
    ps_list   = []
    pb_list   = []
    beta_list = []
    roe_list  = []
    vol_list  = []

    print(f"  Fetching {len(tickers)} tickers for {sector}...")

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info

            pe   = info.get("trailingPE")
            ps   = info.get("priceToSalesTrailing12Months")
            pb   = info.get("priceToBook")
            beta = info.get("beta")
            roe  = info.get("returnOnEquity")

            # Annualised volatility from 1-year daily returns
            hist = yf.Ticker(ticker).history(period="1y")
            if len(hist) > 20:
                daily_returns = hist["Close"].pct_change().dropna()
                vol = float(daily_returns.std() * np.sqrt(252))
            else:
                vol = None

            if pe   and pe   > 0:   pe_list.append(pe)
            if ps   and ps   > 0:   ps_list.append(ps)
            if pb   and pb   > 0:   pb_list.append(pb)
            if beta and beta > 0:   beta_list.append(beta)
            if roe  and roe  > 0:   roe_list.append(roe)
            if vol  and vol  > 0:   vol_list.append(vol)

        except Exception as e:
            print(f"    Warning: could not fetch {ticker}: {e}")

    def median_or_none(lst):
        return round(float(np.median(lst)), 4) if lst else None

    return {
        "pe":         median_or_none(pe_list),
        "ps":         median_or_none(ps_list),
        "pb":         median_or_none(pb_list),
        "beta":       median_or_none(beta_list),
        "roe":        median_or_none(roe_list),
        "volatility": median_or_none(vol_list),
        "sample_size": len(tickers),
    }


def update_benchmarks():
    """
    Fetch sector metrics for all sectors and write to sector_benchmarks.json.
    """
    print("EquityMind — Sector Benchmark Updater")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 50)

    sector_pe         = {}
    sector_ps         = {}
    sector_pb         = {}
    sector_beta       = {}
    sector_volatility = {}
    sector_roe        = {}

    for sector, tickers in SECTOR_TICKERS.items():
        print(f"\n[{sector}]")
        metrics = fetch_sector_metrics(sector, tickers)

        if metrics["pe"]:         sector_pe[sector]         = metrics["pe"]
        if metrics["ps"]:         sector_ps[sector]         = metrics["ps"]
        if metrics["pb"]:         sector_pb[sector]         = metrics["pb"]
        if metrics["beta"]:       sector_beta[sector]       = metrics["beta"]
        if metrics["volatility"]: sector_volatility[sector] = metrics["volatility"]
        if metrics["roe"]:        sector_roe[sector]        = metrics["roe"]

        print(f"  PE={metrics['pe']} | PS={metrics['ps']} | "
              f"Beta={metrics['beta']} | Vol={metrics['volatility']}")

    benchmarks = {
        "updated_at": str(date.today()),
        "source":     "yfinance S&P500 median, computed by scripts/update_benchmarks.py",
        "notes":      "Update quarterly by running: python scripts/update_benchmarks.py",
        "default_pe":        int(round(float(np.median(list(sector_pe.values()))))),
        "default_ps":        int(round(float(np.median(list(sector_ps.values()))))),
        "sector_pe":         sector_pe,
        "sector_ps":         sector_ps,
        "sector_pb":         sector_pb,
        "sector_beta":       sector_beta,
        "sector_volatility": sector_volatility,
        "sector_roe":        sector_roe,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(benchmarks, f, indent=4)

    print(f"\n✓ Benchmarks updated and saved to {OUTPUT_PATH}")
    print(f"  updated_at: {benchmarks['updated_at']}")


if __name__ == "__main__":
    update_benchmarks()