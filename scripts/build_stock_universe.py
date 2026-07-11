"""
scripts/build_stock_universe.py

Builds the candidate stock universe for EquityMind's Layer 2 batch signal
computation — top 200 large-cap US stocks by market cap.

Data source: yfinance only. No web scraping, no third-party constituent
APIs (FMP's legacy S&P 500/Nasdaq constituent endpoints were deprecated
in August 2025 and return 403 for all current accounts).

CANDIDATE_POOL below is a static list of well-known large-cap US tickers
spanning major sectors — a reasonable, broad starting universe, not an
official index constituent list. The script ranks this pool by current
market cap and keeps the top 200.

Usage:
    python scripts/build_stock_universe.py

Output:
    src/quant/data/stock_universe.json
"""

import json
import yfinance as yf
from datetime import date
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "src" / "quant" / "data" / "stock_universe.json"
TOP_N = 250

# ─────────────────────────────────────────────
# Static candidate pool — well-known large-cap US tickers across sectors.
# Not an official index list; the script ranks these by market cap and
# keeps the top TOP_N, so this just needs to be broad enough to cover
# the companies actually worth including.
# ─────────────────────────────────────────────
CANDIDATE_POOL = [
    # Technology / Software / Internet
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "AVGO", "ORCL",
    "CRM", "AMD", "INTU", "IBM", "NOW", "ADBE", "CSCO", "QCOM", "TXN",
    "AMAT", "MU", "PANW", "SNPS", "CDNS", "FTNT", "ANET", "WDAY", "TEAM",
    "DDOG", "SNOW", "NET", "ZS", "CRWD", "PLTR", "UBER", "ABNB", "SPOT",
    "SHOP", "SQ", "PYPL", "MELI", "BKNG", "NFLX", "DIS",
    "APP", "TTD", "DASH", "HOOD", "ROKU", "PINS", "SNAP", "U", "DOCU",
    "OKTA", "TWLO", "HUBS", "BILL", "PCTY", "MNDY", "S", "TEM", "IOT",
    "AXON", "PLTK", "RBLX", "EA", "TTWO", "MSTR", "APPF",

    # Semiconductors / Hardware
    "TSM", "ASML", "LRCX", "KLAC", "ADI", "MCHP", "ON", "MRVL",
    "NXPI", "SWKS", "MPWR", "STX", "WDC", "TER", "ENTG", "COHR", "SMCI",
    "ARM", "GFS",

    # Automotive / EV
    "TSLA", "RIVN", "GM", "F", "TM",

    # Aerospace / Defense
    "BA", "LMT", "RTX", "NOC", "GD", "SPCX",

    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP", "BLK", "SPGI",
    "USB", "PNC", "TFC", "COF", "BK", "V", "MA",
    "ICE", "CME", "MMC", "AON", "AJG", "BRO", "TROW", "STT", "NTRS",
    "AIG", "MET", "PRU", "ALL", "TRV", "PGR", "CB", "HIG", "AFL",
    "SYF", "DFS", "FIS", "FI", "GPN", "JKHY", "SOFI", "AFRM",

    # Healthcare / Pharma
    "JNJ", "LLY", "UNH", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "CI", "ELV", "HUM", "ISRG", "VRTX", "REGN",
    "MRNA", "BSX", "SYK", "EW", "MDT", "ZBH", "BAX", "BDX", "IDXX",
    "IQV", "A", "MTD", "WAT", "RVTY", "ALGN", "DXCM", "PODD", "VTRS",
    "ZTS", "CNC", "MOH",

    # Energy / Utilities
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "WMB",
    "KMI", "OKE", "HAL", "BKR", "FANG", "DVN", "HES", "TRGP", "EXC",
    "ED", "PEG", "SRE", "XEL", "WEC", "ES", "ETR", "FE", "AEE", "CMS",
    "NEE", "DUK", "SO", "AEP",

    # Consumer / Retail
    "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT", "HD",
    "LOW", "TJX", "EL", "CL", "KMB", "GIS", "KHC", "MDLZ",
    "ORLY", "AZO", "ROST", "DG", "DLTR", "YUM", "CMG", "DPZ", "DRI",
    "HLT", "MAR", "RCL", "CCL", "NCLH", "LULU", "ULTA", "TSCO", "BBY",
    "KR", "SYY", "CAG", "HSY", "STZ", "KDP", "MNST", "CLX", "CHD",
    "KVUE",

    # Industrials
    "HON", "UPS", "CAT", "DE", "MMM", "GE", "UNP", "EMR", "ETN", "ITW",
    "ADP", "PAYX", "CTAS", "FAST", "PH", "ROK", "DOV", "IR", "CMI",
    "PCAR", "XYL", "AME", "GEV", "TT", "CARR", "OTIS", "FDX", "NSC",
    "CSX", "WM", "RSG", "URI", "EFX",

    # Communication / Media
    "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "PARA", "WBD", "LYV", "FOXA", "MTCH",

    # Real Estate / REITs
    "AMT", "PLD", "EQIX", "SPG", "O",
    "PSA", "DLR", "VICI", "WELL", "ARE", "AVB", "EQR", "MAA", "ESS",
    "INVH", "VTR", "CPT", "KIM", "REG", "BXP", "HST",

    # Recent large-cap IPOs / notable growth names
    "COIN", "RDDT", "CRCL", "MDB", "GTLB", "PATH", "ZIP", "WRBY", "AMPL",
    "CFLT", "CPNG", "COMP", "CLOV", "AI",
]


def rank_by_market_cap(symbols: list[str], top_n: int) -> list[dict]:
    """
    Fetch market cap and company name for each ticker via yfinance, and
    return the top_n by market cap, sorted descending. Skips any ticker
    that fails to fetch or has no market cap data.

    Company name is captured here (not just market cap) so this same
    single pass of yfinance calls can also produce a ticker<->company-name
    lookup table — used as a fallback when extract_parameters' LLM-based
    name-to-ticker conversion fails (e.g. for less-famous companies).
    """
    ranked = []
    seen = set()
    for i, symbol in enumerate(symbols):
        if symbol in seen:
            continue
        seen.add(symbol)
        try:
            info = yf.Ticker(symbol).info
            market_cap = info.get("marketCap")
            company_name = info.get("longName") or info.get("shortName")
            if market_cap:
                ranked.append({
                    "symbol": symbol,
                    "market_cap": market_cap,
                    "company_name": company_name,
                })
        except Exception as e:
            print(f"    Skipping {symbol}: {e}")

        if (i + 1) % 50 == 0:
            print(f"    ...processed {i + 1}/{len(symbols)}")

    ranked.sort(key=lambda x: x["market_cap"], reverse=True)
    return ranked[:top_n]


def build_stock_universe():
    print("EquityMind — Stock Universe Builder")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 50)
    print(f"\nCandidate pool: {len(CANDIDATE_POOL)} tickers")
    print("Fetching market caps and ranking...")

    top_ranked = rank_by_market_cap(CANDIDATE_POOL, TOP_N)

    universe = [entry["symbol"] for entry in top_ranked]
    details = {
        entry["symbol"]: {
            "market_cap": entry["market_cap"],
            "company_name": entry["company_name"],
        }
        for entry in top_ranked
    }

    output = {
        "updated_at": str(date.today()),
        "description": f"Top {TOP_N} US large-cap stocks by market cap, "
                        f"selected from a static candidate pool of "
                        f"{len(CANDIDATE_POOL)} well-known tickers.",
        "total_tickers": len(universe),
        "tickers": universe,
        "details": details,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ stock_universe.json written — {len(universe)} tickers")


if __name__ == "__main__":
    build_stock_universe()
