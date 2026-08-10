"""
scripts/build_stock_universe.py

Ranks CANDIDATE_POOL by market cap (yfinance), keeps the top TOP_N,
and writes ticker/market_cap/company_name to the stock_universe table.
Candidates must report in USD and file 10-K — see the note above
CANDIDATE_POOL. Tickers no longer in the top TOP_N are deleted
(current-snapshot table, not accumulating history).

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

from scripts.filing_check import is_usd_reporter, files_20f_only
from config import DATABASE_URL

TOP_N = 250

# Candidate pool, ranked by market cap. Two requirements: USD financial
# reporting (so F-Score comparisons are meaningful) and 10-K filing (so
# Item 1 is available for overview embeddings). Foreign private issuers
# file 20-F, not 10-K, and are dropped by filter_domestic_filers().
# Known cases already removed from this list: ARM, GFS, MNDY.
# Dual-class listings are represented once, by the voting share
# (GOOGL, not GOOG): both classes share one 10-K and one set of
# financials, so a second entry would duplicate every signal and
# occupy a slot in Discovery's results.
CANDIDATE_POOL = [
    # Technology / Software / Internet
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL",
    "CRM", "AMD", "INTU", "IBM", "NOW", "ADBE", "CSCO", "QCOM", "TXN",
    "AMAT", "MU", "PANW", "SNPS", "CDNS", "FTNT", "ANET", "WDAY", "TEAM",
    "DDOG", "SNOW", "NET", "ZS", "CRWD", "PLTR", "UBER", "ABNB",
    "SHOP", "XYZ", "PYPL", "MELI", "BKNG", "NFLX", "DIS",
    "APP", "TTD", "DASH", "HOOD", "ROKU", "PINS", "SNAP", "U", "DOCU",
    "OKTA", "TWLO", "HUBS", "BILL", "PCTY", "S", "TEM", "IOT",
    "AXON", "PLTK", "RBLX", "EA", "TTWO", "MSTR", "APPF",

    # Semiconductors / Hardware
    "LRCX", "KLAC", "ADI", "MCHP", "ON", "MRVL",
    "NXPI", "SWKS", "MPWR", "STX", "WDC", "TER", "ENTG", "COHR", "SMCI",

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
    ranking here — see filter_usd_reporters(), filter_domestic_filers()
    and rank_by_market_cap().
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
    for i, c in enumerate(candidates):
        if is_usd_reporter(c["symbol"]):
            filtered.append(c)
        else:
            print(f"    Skipping {c['symbol']}: non-USD financial reporting")

        if (i + 1) % 50 == 0:
            print(f"    ...checked {i + 1}/{len(candidates)}")

    return filtered


def filter_domestic_filers(candidates: list[dict]) -> list[dict]:
    """Removes foreign private issuers, which file 20-F rather than 10-K
    and so have no Item 1 business description for overview embeddings.
    Runs after filter_usd_reporters() because this one hits SEC EDGAR and
    is the slower of the two."""
    filtered = []
    for i, c in enumerate(candidates):
        if files_20f_only(c["symbol"]):
            print(f"    Skipping {c['symbol']}: foreign private issuer (files 20-F)")
        else:
            filtered.append(c)

        if (i + 1) % 50 == 0:
            print(f"    ...checked {i + 1}/{len(candidates)}")

    return filtered


def rank_by_market_cap(candidates: list[dict], top_n: int) -> list[dict]:
    """
    Sorts candidates by market cap descending and returns the top_n.
    Pure sort-and-truncate — no fetching, no filtering (see
    fetch_candidate_data(), filter_usd_reporters() and
    filter_domestic_filers() for those steps).
    """
    ranked = sorted(candidates, key=lambda x: x["market_cap"], reverse=True)
    return ranked[:top_n]


def build_stock_universe():
    print("EquityMind — Stock Universe Builder")
    print("=" * 50)
    print(f"\nCandidate pool: {len(CANDIDATE_POOL)} tickers")
    print("Fetching market caps...")

    candidates = fetch_candidate_data(CANDIDATE_POOL)

    print("Checking financial reporting currency...")
    candidates = filter_usd_reporters(candidates)

    print("Checking annual report form (10-K vs 20-F)...")
    candidates = filter_domestic_filers(candidates)
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
