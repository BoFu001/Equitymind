"""
scripts/update_peer_groups.py

Peer Group Benchmark Updater for EquityMind Layer 2 Quant Engine.

Data sources:
    Primary:  Financial Modeling Prep API — /stable/stock-peers
              Fetches real peer companies per ticker, computes median P/E and P/B
    Fallback (P/E only): Damodaran industry dataset (NYU Stern, updated annually)
              Used when FMP peer group is unavailable or empty.
              Damodaran does not publish industry P/B, so P/B has no
              equivalent fallback tier — it drops straight to the global default.

Usage:
    python scripts/update_peer_groups.py

Run quarterly to keep benchmarks current.
Requires: requests, yfinance, pandas, xlrd, numpy, python-dotenv
"""

import os
import json
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

FMP_API_KEY  = os.getenv("FMP_API_KEY")
OUTPUT_PATH  = Path(__file__).parent.parent / "src" / "quant" / "data" / "peer_benchmarks.json"

# ─────────────────────────────────────────────
# Companies to build peer benchmarks for.
# Loaded from stock_universe.json (built by build_stock_universe.py) so
# this list always stays in sync with the current candidate stock pool,
# instead of maintaining a separate hardcoded list here.
# ─────────────────────────────────────────────
def _load_target_tickers() -> list[str]:
    universe_path = Path(__file__).parent.parent / "src" / "quant" / "data" / "stock_universe.json"
    with open(universe_path, "r") as f:
        universe = json.load(f)
    return universe["tickers"]


TARGET_TICKERS = _load_target_tickers()

# ─────────────────────────────────────────────
# Damodaran industry P/E mapping
# Maps yfinance industry string → Damodaran industry name
# Note: Damodaran does not publish industry P/B, only P/E — this
# fallback tier applies to P/E only.
# ─────────────────────────────────────────────
INDUSTRY_TO_DAMODARAN = {
    "Internet Content & Information": "Software (Internet)",
    "Software - Application":         "Software (System & Application)",
    "Software - Infrastructure":      "Software (System & Application)",
    "Semiconductors":                 "Semiconductor",
    "Semiconductor Equipment":        "Semiconductor Equip",
    "Consumer Electronics":           "Electronics (Consumer & Office)",
    "Internet Retail":                "Retail (General)",
    "Auto Manufacturers":             "Auto & Truck",
    "Telecom Services":               "Telecom. Services",
    "Entertainment":                  "Entertainment",
    "Aerospace & Defense":            "Aerospace/Defense",
    "Drug Manufacturers - General":   "Drugs (Pharmaceutical)",
    "Biotechnology":                  "Drugs (Biotechnology)",
    "Banks - Diversified":            "Bank (Money Center)",
    "Oil & Gas Integrated":          "Oil/Gas (Integrated)",
    "Discount Stores":                "Retail (General)",
    "Household & Personal Products":  "Household Products",
    "Beverages - Non-Alcoholic":      "Beverage (Soft)",
}


def fetch_damodaran_pe() -> dict:
    """
    Download Damodaran industry P/E dataset and return as dict:
    {industry_name: forward_pe}
    """
    print("  Fetching Damodaran industry P/E data...")
    url = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/pedata.xls"
    df  = pd.read_excel(url, sheet_name="Industry Averages", header=6)
    df.columns = ["industry", "url", "skip", "num_firms", "pct_loss",
                  "current_pe", "trailing_pe", "forward_pe", "agg_pe", "peg"]
    df = df[["industry", "forward_pe"]].dropna(subset=["industry"])
    df = df[df["industry"] != "Industry Name"]
    df = df[pd.to_numeric(df["forward_pe"], errors="coerce").notnull()]
    result = {row["industry"]: round(float(row["forward_pe"]), 2)
              for _, row in df.iterrows()}
    print(f"  Loaded {len(result)} Damodaran industries")
    return result


def fetch_peer_median_pe_pb(ticker: str) -> tuple[float | None, float | None]:
    """
    Fetch peer group from FMP, compute median P/E and median P/B of peers
    in a single pass over the same peer list — P/B is essentially free to
    add here since we already fetch each peer's yfinance info for P/E.

    Returns (median_pe, median_pb), either element None if unavailable.
    """
    url = f"https://financialmodelingprep.com/stable/stock-peers?symbol={ticker}&apikey={FMP_API_KEY}"
    r   = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None, None

    peers = json.loads(r.text)
    if not peers:
        return None, None

    pe_list = []
    pb_list = []
    for peer in peers:
        symbol = peer.get("symbol")
        if not symbol or symbol == ticker:
            continue
        try:
            info = yf.Ticker(symbol).info
            pe = info.get("trailingPE")
            pb = info.get("priceToBook")
            # Filter out extreme values (loss-making or bubble stocks)
            if pe and 3 < pe < 150:
                pe_list.append(pe)
            # P/B filter: negative/zero excluded (negative equity is a
            # distress signal, not a valid peer comparison point); very
            # high P/B (>50) usually reflects data errors or extreme
            # asset-light outliers rather than a meaningful peer level.
            if pb and 0 < pb < 50:
                pb_list.append(pb)
        except Exception:
            continue

    median_pe = round(float(np.median(pe_list)), 2) if pe_list else None
    median_pb = round(float(np.median(pb_list)), 2) if pb_list else None

    return median_pe, median_pb


def get_damodaran_fallback(ticker: str, damodaran_data: dict) -> float | None:
    """
    Look up Damodaran industry P/E for a ticker using yfinance industry field.
    Returns None if no mapping found. P/E only — Damodaran has no P/B fallback.
    """
    try:
        industry = yf.Ticker(ticker).info.get("industry", "")
        damodaran_industry = INDUSTRY_TO_DAMODARAN.get(industry)
        if damodaran_industry and damodaran_industry in damodaran_data:
            return damodaran_data[damodaran_industry]
    except Exception:
        pass
    return None


def update_peer_benchmarks():
    """
    Main function: build peer_benchmarks.json from FMP + Damodaran.

    P/E has three tiers: FMP peer group -> Damodaran industry -> global default.
    P/B has two tiers only: FMP peer group -> global default (no industry
    fallback dataset exists for P/B).
    """
    print("EquityMind — Peer Group Benchmark Updater")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 50)

    # Load Damodaran data once
    damodaran_data = fetch_damodaran_pe()

    peer_benchmarks = {}
    # Global fallback values, sourced from real S&P 500 aggregate data
    # (GuruFocus / S&P Dow Jones Indices) rather than an arbitrary guess.
    # Update these figures whenever this script is re-run (quarterly) by
    # checking https://www.gurufocus.com/economic_indicators/57/sp-500-pe-ratio
    # and https://www.gurufocus.com/economic_indicators/4240/sp-500-price-to-book-value
    default_pe = 25.54   # S&P 500 P/E, GuruFocus, as of 2026-07-06
    default_pb = 5.44    # S&P 500 P/B, GuruFocus/S&P Dow Jones Indices, as of 2026-01-23

    for ticker in TARGET_TICKERS:
        print(f"\n[{ticker}]")

        # ── P/E and P/B: primary source is FMP peer group (single fetch) ──
        peer_pe, peer_pb = fetch_peer_median_pe_pb(ticker)

        # ── P/E fallback chain: FMP -> Damodaran -> default ──
        if peer_pe:
            pe_source = "fmp_peer_group"
            pe        = peer_pe
            print(f"  FMP peer median P/E: {pe}")
        else:
            damodaran_pe = get_damodaran_fallback(ticker, damodaran_data)
            if damodaran_pe:
                pe_source = "damodaran"
                pe        = damodaran_pe
                print(f"  Damodaran fallback P/E: {pe}")
            else:
                pe_source = "default"
                pe        = default_pe
                print(f"  Default fallback P/E: {pe}")

        # ── P/B fallback chain: FMP -> default (no industry dataset available) ──
        if peer_pb:
            pb_source = "fmp_peer_group"
            pb        = peer_pb
            print(f"  FMP peer median P/B: {pb}")
        else:
            pb_source = "default"
            pb        = default_pb
            print(f"  Default fallback P/B: {pb}")

        peer_benchmarks[ticker] = {
            "benchmark_pe":    pe,
            "pe_source":       pe_source,
            "benchmark_pb":    pb,
            "pb_source":       pb_source,
        }

    output = {
        "updated_at": str(date.today()),
        "source":     "FMP stock-peers API + Damodaran industry dataset (P/E only)",
        "notes":      "Run quarterly: python scripts/update_peer_groups.py",
        "default_pe": default_pe,
        "default_pb": default_pb,
        "benchmarks": peer_benchmarks,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n✓ peer_benchmarks.json updated — {len(peer_benchmarks)} tickers")
    fmp_pe_count = sum(1 for v in peer_benchmarks.values() if v["pe_source"] == "fmp_peer_group")
    dam_pe_count = sum(1 for v in peer_benchmarks.values() if v["pe_source"] == "damodaran")
    def_pe_count = sum(1 for v in peer_benchmarks.values() if v["pe_source"] == "default")
    fmp_pb_count = sum(1 for v in peer_benchmarks.values() if v["pb_source"] == "fmp_peer_group")
    def_pb_count = sum(1 for v in peer_benchmarks.values() if v["pb_source"] == "default")
    print(f"  P/E — FMP peer group: {fmp_pe_count} | Damodaran: {dam_pe_count} | Default: {def_pe_count}")
    print(f"  P/B — FMP peer group: {fmp_pb_count} | Default: {def_pb_count}")


if __name__ == "__main__":
    update_peer_benchmarks()
