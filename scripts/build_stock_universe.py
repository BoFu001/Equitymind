"""
scripts/build_stock_universe.py

Ranks CANDIDATE_POOL by market cap (yfinance), keeps the top TOP_N
USD-reporting companies, and writes ticker/market_cap/company_name to
the stock_universe table. Tickers no longer in the top TOP_N are
deleted (current-snapshot table, not accumulating history).

common_name and peers are written separately by
update_common_names.py and update_peer_groups.py.

Usage:
    python scripts/build_stock_universe.py
"""

import sys
import psycopg2
import yfinance as yf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.currency_check import is_usd_reporter
from config import DATABASE_URL

TOP_N = 250

# Candidate pool, ranked by market cap; only USD financial reporters pass.
CANDIDATE_POOL = [
    # Technology / Software / Internet
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN", "AVGO", "ORCL",
    "CRM", "AMD", "INTU", "IBM", "NOW", "ADBE", "CSCO", "QCOM", "TXN",
    "AMAT", "MU", "PANW", "SNPS", "CDNS", "FTNT", "ANET", "WDAY", "TEAM",
    "DDOG", "SNOW", "NET", "ZS", "CRWD", "PLTR", "UBER", "ABNB",
    "SHOP", "XYZ", "PYPL", "MELI", "BKNG", "NFLX", "DIS",
    "APP", "TTD", "DASH", "HOOD", "ROKU", "PINS", "SNAP", "U", "DOCU",
    "OKTA", "TWLO", "HUBS", "BILL", "PCTY", "MNDY", "S", "TEM", "IOT",
    "AXON", "PLTK", "RBLX", "EA", "TTWO", "MSTR", "APPF",

    # Semiconductors / Hardware
    "LRCX", "KLAC", "ADI", "MCHP", "ON", "MRVL",
    "NXPI", "SWKS", "MPWR", "STX", "WDC", "TER", "ENTG", "COHR", "SMCI",
    "ARM", "GFS",

    # Automotive / EV
    "TSLA", "RIVN", "GM", "F",

    # Aerospace / Defense
    "BA", "LMT", "RTX", "NOC", "GD", "SPCX",

    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP", "BLK", "SPGI",
    "USB", "PNC", "TFC", "COF", "V", "MA",
    "ICE", "CME", "MRSH", "AON", "AJG", "BRO", "TROW", "STT", "NTRS",
    "AIG", "MET", "PRU", "ALL", "TRV", "PGR", "CB", "HIG", "AFL",
    "SYF", "FIS", "GPN", "JKHY", "SOFI", "AFRM",

    # Healthcare / Pharma
    "JNJ", "LLY", "UNH", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "CI", "ELV", "HUM", "ISRG", "VRTX", "REGN",
    "MRNA", "BSX", "SYK", "EW", "MDT", "ZBH", "BAX", "BDX", "IDXX",
    "IQV", "A", "MTD", "WAT", "RVTY", "ALGN", "DXCM", "PODD", "VTRS",
    "ZTS", "CNC", "MOH",

    # Energy / Utilities
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "WMB",
    "KMI", "OKE", "HAL", "BKR", "FANG", "DVN", "TRGP", "EXC",
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
    "WBD", "LYV", "FOXA", "MTCH",

    # Real Estate / REITs
    "AMT", "PLD", "EQIX", "SPG", "O",
    "PSA", "DLR", "VICI", "WELL", "ARE", "AVB", "EQR", "MAA", "ESS",
    "INVH", "VTR", "CPT", "KIM", "REG", "BXP", "HST",

    # Recent large-cap IPOs / notable growth names
    "COIN", "RDDT", "CRCL", "MDB", "GTLB", "PATH", "ZIP", "WRBY", "AMPL",
    "CPNG", "COMP", "CLOV", "AI",
]


def fetch_candidate_data(symbols: list[str]) -> list[dict]:
    """
    Fetches market cap and company name for each ticker via yfinance.
    Skips tickers with no market cap data. No currency filtering or
    ranking here — see filter_usd_reporters() and rank_by_market_cap().
    """
    fetched = []
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
                fetched.append({
                    "symbol": symbol,
                    "market_cap": market_cap,
                    "company_name": company_name,
                })
        except Exception as e:
            print(f"    Skipping {symbol}: {e}")

        if (i + 1) % 50 == 0:
            print(f"    ...processed {i + 1}/{len(symbols)}")

    return fetched


def filter_usd_reporters(candidates: list[dict]) -> list[dict]:
    """Removes candidates not reporting financials in USD."""
    filtered = []
    for c in candidates:
        if is_usd_reporter(c["symbol"]):
            filtered.append(c)
        else:
            print(f"    Skipping {c['symbol']}: non-USD financial reporting")
    return filtered


def rank_by_market_cap(candidates: list[dict], top_n: int) -> list[dict]:
    """
    Sorts candidates by market cap descending and returns the top_n.
    Pure sort-and-truncate — no fetching, no filtering (see
    fetch_candidate_data() and filter_usd_reporters() for those steps).
    """
    ranked = sorted(candidates, key=lambda x: x["market_cap"], reverse=True)
    return ranked[:top_n]


def build_stock_universe():
    print("EquityMind — Stock Universe Builder")
    print("=" * 50)
    print(f"\nCandidate pool: {len(CANDIDATE_POOL)} tickers")
    print("Fetching market caps...")

    candidates = fetch_candidate_data(CANDIDATE_POOL)
    candidates = filter_usd_reporters(candidates)
    top_ranked = rank_by_market_cap(candidates, TOP_N)
    tickers = [entry["symbol"] for entry in top_ranked]

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Snapshot before/after against the actual DB state, so a failed insert isn't misreported as "arrived".
    cursor.execute("SELECT ticker FROM stock_universe")
    previous_tickers = {row[0] for row in cursor.fetchall()}

    print("\nRemoving tickers no longer in the universe...")
    cursor.execute("DELETE FROM stock_universe WHERE ticker != ALL(%s)", (tickers,))
    removed = cursor.rowcount
    conn.commit()

    print("Writing ranked universe to stock_universe table...")
    for entry in top_ranked:
        cursor.execute("""
            INSERT INTO stock_universe (ticker, market_cap, company_name)
            VALUES (%s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
                market_cap = EXCLUDED.market_cap,
                company_name = EXCLUDED.company_name;
        """, (entry["symbol"], entry["market_cap"], entry["company_name"]))
    conn.commit()

    cursor.execute("SELECT ticker FROM stock_universe")
    current_tickers = {row[0] for row in cursor.fetchall()}
    departed = sorted(previous_tickers - current_tickers)
    arrived = sorted(current_tickers - previous_tickers)
    print(f"  Removed {removed} ticker(s) no longer in the top {TOP_N}.")
    print(f"  Departed: {departed if departed else 'none'}")
    print(f"  Arrived: {arrived if arrived else 'none'}\n")

    cursor.close()
    conn.close()

    print(f"\n✓ stock_universe table updated — {len(tickers)} tickers")


if __name__ == "__main__":
    build_stock_universe()
