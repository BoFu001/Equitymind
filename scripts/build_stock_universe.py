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
import time
import psycopg2
import yfinance as yf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.filing_check import is_usd_reporter, files_20f_only
from config import DATABASE_URL, PIPELINE_THROTTLE_SECONDS

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


# Retry exists to catch the occasional dropped request, not to outlast a
# rate-limit window — those run for minutes (17 on 2026-08-17, ~6 on
# 2026-08-14), so no delay short enough to be worth waiting would clear
# one. The stored-value fallback is what actually protects the universe;
# this just avoids using it for a blip.
RETRY_DELAY_SECONDS = 3

# Once this many tickers in a row have failed even after a retry, the
# limit is clearly sustained and retrying the remaining candidates only
# adds RETRY_DELAY_SECONDS each to a run that will fall back regardless.
# Retries stop for the rest of the run; the fallback still applies.
RETRY_GIVEUP_AFTER = 10


def _load_stored_market_caps(cursor) -> dict[str, tuple[int, str]]:
    """The universe's own last-known values, used when yfinance will not
    answer. Every current member has a market cap, so this covers every
    ticker that could be deleted by a failed run."""
    cursor.execute("""
        SELECT ticker, market_cap, company_name
        FROM stock_universe
        WHERE market_cap IS NOT NULL
    """)
    return {row[0]: (row[1], row[2]) for row in cursor.fetchall()}


def _fetch_one(symbol: str) -> tuple[int, str] | None:
    """One attempt. Returns None if no usable market cap came back.

    Rate limiting shows up two ways — an exception, and a 200 response
    with no marketCap field — so both are treated as the same failure
    rather than only the first being noticed."""
    try:
        info = yf.Ticker(symbol).info
    except Exception as e:
        print(f"    {symbol}: {e}", flush=True)
        return None

    market_cap = info.get("marketCap")
    company_name = info.get("longName") or info.get("shortName")

    if market_cap:
        return market_cap, company_name
    else:
        print(f"    {symbol}: response contained no marketCap", flush=True)
        return None


def fetch_candidate_data(
    symbols: list[str],
    stored: dict[str, tuple[int, str]],
) -> tuple[list[dict], list[str], list[str]]:
    """
    Fetches market cap and company name for each ticker via yfinance,
    retrying once and falling back to the stored value.

    A failed fetch used to drop the ticker from the candidate list
    entirely, which rank_by_market_cap() could not distinguish from a
    company too small to make the top TOP_N — so the universe write
    deleted it, along with its overview, tags and peer group. On
    2026-08-17 a rate-limit window that began in this script's first
    second removed AAPL, MSFT, NVDA, GOOGL, META and nine others from
    the universe this way. Falling back to the last known market cap
    treats a failure as "unknown, assume unchanged" rather than "worth
    nothing": a company's market cap does not move enough in a day to
    change its ranking materially, and every current member has a
    stored value, so no existing member can now be deleted by a fetch
    failure alone.

    Returns (candidates, used_stored, no_data). The two extra lists let
    the caller report what happened — a silently short list was what
    made the original failure invisible.
    """
    fetched = []
    used_stored = []
    no_data = []
    seen = set()
    retries_enabled = True
    consecutive_failures = 0

    for i, symbol in enumerate(symbols):
        if symbol in seen:
            continue
        seen.add(symbol)

        result = _fetch_one(symbol)

        if result is None and retries_enabled:
            time.sleep(RETRY_DELAY_SECONDS)
            result = _fetch_one(symbol)

            if result is None:
                consecutive_failures += 1
                if consecutive_failures >= RETRY_GIVEUP_AFTER:
                    retries_enabled = False
                    print(f"    {RETRY_GIVEUP_AFTER} consecutive retries failed — "
                          f"treating the limit as sustained and skipping further "
                          f"retries; stored values still apply", flush=True)
            else:
                consecutive_failures = 0

        if result is not None:
            market_cap, company_name = result
            fetched.append({
                "symbol": symbol,
                "market_cap": market_cap,
                "company_name": company_name,
            })
        elif symbol in stored:
            market_cap, company_name = stored[symbol]
            print(f"    {symbol}: using stored market cap {market_cap:,d}", flush=True)
            fetched.append({
                "symbol": symbol,
                "market_cap": market_cap,
                "company_name": company_name,
            })
            used_stored.append(symbol)
        else:
            # Never been in the universe, so there is nothing to fall
            # back to. Dropping it is safe: it cannot delete data that
            # was never written.
            print(f"    {symbol}: no data and no stored value — excluded", flush=True)
            no_data.append(symbol)

        if PIPELINE_THROTTLE_SECONDS > 0:
            time.sleep(PIPELINE_THROTTLE_SECONDS)

        if (i + 1) % 50 == 0:
            print(f"    ...processed {i + 1}/{len(symbols)}", flush=True)

    return fetched, used_stored, no_data


def filter_usd_reporters(candidates: list[dict], members: set[str]) -> list[dict]:
    """Removes candidates not reporting financials in USD.

    Current universe members skip the check. is_usd_reporter() calls
    yfinance and returns False on any failure — deliberately, since an
    unverifiable new candidate should not be admitted — but for a company
    already in the universe that same False means "rate limited" is read
    as "not a USD reporter", and the ticker is dropped and then deleted
    along with its overview, tags and peers. It passed this check when it
    was admitted, and reporting currency does not change overnight, so
    the stored membership is better evidence than a call that may not
    answer. New candidates are still checked strictly."""
    filtered = []
    for i, c in enumerate(candidates):
        if c["symbol"] in members:
            filtered.append(c)
        elif is_usd_reporter(c["symbol"]):
            filtered.append(c)
        else:
            print(f"    Skipping {c['symbol']}: non-USD financial reporting", flush=True)

        if (i + 1) % 50 == 0:
            print(f"    ...checked {i + 1}/{len(candidates)}", flush=True)

    return filtered


def filter_domestic_filers(candidates: list[dict], members: set[str]) -> list[dict]:
    """Removes foreign private issuers, which file 20-F rather than 10-K
    and so have no Item 1 business description. Runs after
    filter_usd_reporters() because this one hits SEC EDGAR and is the
    slower of the two.

    Current members skip the check, for the reason given in
    filter_usd_reporters() and for cost: this is throttled per request,
    so re-verifying 250 members daily is most of the script's SEC
    traffic. The one direction that would matter — a member reverting
    from 10-K to 20-F — is rare, and would surface anyway as a missing
    Item 1 in update_stock_overviews.py rather than silently."""
    filtered = []
    for i, c in enumerate(candidates):
        if c["symbol"] in members:
            filtered.append(c)
        elif files_20f_only(c["symbol"]):
            print(f"    Skipping {c['symbol']}: foreign private issuer (files 20-F)", flush=True)
        else:
            filtered.append(c)

        if (i + 1) % 50 == 0:
            print(f"    ...checked {i + 1}/{len(candidates)}", flush=True)

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
    print("EquityMind — Stock Universe Builder", flush=True)
    print("=" * 50, flush=True)

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    stored = _load_stored_market_caps(cursor)

    print(f"\nCandidate pool: {len(CANDIDATE_POOL)} tickers", flush=True)
    print(f"Stored market caps available as fallback: {len(stored)}", flush=True)
    print("Fetching market caps...", flush=True)

    candidates, used_stored, no_data = fetch_candidate_data(CANDIDATE_POOL, stored)
    fetched_count = len(candidates)

    # Membership is read before any write, so it reflects yesterday's
    # universe — which is what "already verified" means here.
    members = set(stored)

    print("Checking financial reporting currency...", flush=True)
    candidates = filter_usd_reporters(candidates, members)

    print("Checking annual report form (10-K vs 20-F)...", flush=True)
    candidates = filter_domestic_filers(candidates, members)
    top_ranked = rank_by_market_cap(candidates, TOP_N)
    tickers = [entry["symbol"] for entry in top_ranked]

    # Snapshot before/after against the actual DB state, so a failed insert isn't misreported as "arrived".
    cursor.execute("SELECT ticker FROM stock_universe")
    previous_tickers = {row[0] for row in cursor.fetchall()}

    print("\nRemoving tickers no longer in the universe...", flush=True)
    cursor.execute("DELETE FROM stock_universe WHERE ticker != ALL(%s)", (tickers,))
    removed = cursor.rowcount
    conn.commit()

    print("Writing ranked universe to stock_universe table...", flush=True)
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
    print(f"  Removed {removed} ticker(s) no longer in the top {TOP_N}.", flush=True)
    print(f"  Departed: {departed if departed else 'none'}", flush=True)
    print(f"  Arrived: {arrived if arrived else 'none'}\n", flush=True)

    cursor.close()
    conn.close()

    # Market-cap sourcing summary, counted against the fetch step rather
    # than the post-filter list, which is a different population.
    # Printed even when empty, so its absence from a log means the
    # section did not run rather than that nothing went wrong.
    print(f"  Market caps fresh from yfinance: {fetched_count - len(used_stored)}", flush=True)
    if used_stored:
        print(f"  Used stored market cap for {len(used_stored)}: {used_stored}", flush=True)
        print("    (yfinance would not answer — ranked on their last known value)", flush=True)
    else:
        print("  Used stored market cap for 0", flush=True)
    if no_data:
        print(f"  No data and no stored value for {len(no_data)}: {no_data}", flush=True)
        print("    (excluded from ranking — none were universe members)", flush=True)
    else:
        print("  No data and no stored value for 0", flush=True)

    print(f"\n✓ stock_universe table updated — {len(tickers)} tickers", flush=True)


if __name__ == "__main__":
    build_stock_universe()
