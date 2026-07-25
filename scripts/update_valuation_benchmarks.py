"""
scripts/update_valuation_benchmarks.py

Valuation Benchmark Updater for EquityMind Layer 2 Quant Engine.

Reads peer group IDENTITY from peer_groups.json (see
update_peer_groups.py — the only step in this pipeline bound by FMP's
free-tier quota) and computes each ticker's benchmark P/E and P/B —
the median P/E and P/B of that ticker's peers — using yfinance only.
No FMP calls happen here, so this script can be run as often as
desired (e.g. daily) without touching any quota, unlike the old
combined update_peer_groups.py this replaces.

Three-tier degradation for P/E, matching valuation_signal.py's
existing tiers:
    1. FMP peer group (from peer_groups.json) — median P/E of named,
       business-similar peers. Most reliable.
    2. Damodaran industry average (NYU Stern, updated annually) — used
       when the ticker has no FMP peer group. Industry-level only, not
       company-specific — see INDUSTRY_TO_DAMODARAN mapping below.
    3. Global S&P 500 default — used when neither of the above is
       available.
P/B only has two tiers (FMP peer group -> global default) — Damodaran
does not publish industry P/B.

peers_used (the subset of FMP's peer list that actually contributed a
valid P/E/P/B, after filtering out missing/extreme values) is stored
alongside the benchmark numbers — this lets format_valuation() name
the actual companies a stock was compared against (e.g. "vs Microsoft,
Alphabet, Meta"), not just an anonymous "peer average" number.

Usage:
    python scripts/update_valuation_benchmarks.py

Run daily (or as often as desired) — no FMP quota constraint.
Requires: yfinance, pandas, xlrd, numpy, python-dotenv
"""

import json
import sys
from datetime import date
from pathlib import Path

import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

from scripts.currency_check import is_usd_reporter

OUTPUT_PATH = Path(__file__).parent.parent / "src" / "quant" / "data" / "valuation_benchmarks.json"
PEER_GROUPS_PATH = Path(__file__).parent.parent / "src" / "quant" / "data" / "peer_groups.json"


def _load_peer_groups() -> dict:
    with open(PEER_GROUPS_PATH, "r") as f:
        data = json.load(f)
    return data["peer_groups"]


# ─────────────────────────────────────────────
# Damodaran industry P/E mapping
# Maps yfinance industry string -> Damodaran industry name
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
    df = pd.read_excel(url, sheet_name="Industry Averages", header=6)
    df.columns = ["industry", "url", "skip", "num_firms", "pct_loss",
                  "current_pe", "trailing_pe", "forward_pe", "agg_pe", "peg"]
    df = df[["industry", "forward_pe"]].dropna(subset=["industry"])
    df = df[df["industry"] != "Industry Name"]
    df = df[pd.to_numeric(df["forward_pe"], errors="coerce").notnull()]
    result = {row["industry"]: round(float(row["forward_pe"]), 2)
              for _, row in df.iterrows()}
    print(f"  Loaded {len(result)} Damodaran industries")
    return result


def _fetch_peer_info(symbol: str, max_retries: int = 2) -> dict | None:
    """
    Fetches yfinance .info for one peer, retrying on failure or an
    empty/incomplete result. Observed in practice (2026-07-24): running
    ~2500 yfinance calls back-to-back (250 tickers x ~10 peers each,
    with overlap) triggers intermittent empty/rate-limited responses —
    the SAME ticker succeeds on a standalone call but silently returns
    an incomplete dict under batch load. Retrying with a short delay
    resolved this in testing (48% FMP-tier hit rate on one run vs 70%
    on a retry of the same data, with no code change — confirming this
    is transient, not a real data gap).
    """
    for attempt in range(max_retries + 1):
        try:
            info = yf.Ticker(symbol).info
            if info and (info.get("trailingPE") or info.get("priceToBook")):
                return info
        except Exception:
            pass
        if attempt < max_retries:
            time.sleep(0.5)
    return None


def compute_peer_median_pe_pb(peer_symbols: list[str], ticker: str) -> tuple[float | None, float | None, list[str]]:
    """
    Given a list of peer ticker symbols (from peer_groups.json), fetch
    each peer's P/E and P/B via yfinance and compute medians.

    Returns (median_pe, median_pb, peers_used) — peers_used is the
    subset of peer_symbols that actually had a valid P/E or P/B after
    filtering (not every peer necessarily contributes; some may lack
    data or get filtered as extreme values). This is what
    format_valuation() will show the user as "compared against: X, Y, Z".
    """
    pe_list = []
    pb_list = []
    peers_used = set()

    for symbol in peer_symbols:
        if symbol == ticker:
            continue
        if not is_usd_reporter(symbol):
            # A peer that reports financials in a non-USD currency
            # (e.g. a foreign ADR) is excluded here for methodological
            # rigor — even though P/E is a dimensionless ratio and not
            # subject to the same unit-mixing error as raw financial
            # figures (see financial_history's currency issue), cross-
            # market valuation levels can differ systematically due to
            # country risk premia, not just business similarity.
            continue
        info = _fetch_peer_info(symbol)
        if info is None:
            continue
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        # Defensive type check: yfinance has, in practice, occasionally
        # returned a non-numeric value for these fields (e.g. the
        # string "Infinity" for BILL, whose EPS was near zero — a
        # genuine mathematical edge case of the P/E ratio, not a data
        # error) — without this check, a single such ticker crashes
        # the entire batch run instead of just being skipped.
        if not isinstance(pe, (int, float)):
            pe = None
        if not isinstance(pb, (int, float)):
            pb = None
        # Filter out extreme values (loss-making or bubble stocks)
        if pe and 3 < pe < 150:
            pe_list.append(pe)
            peers_used.add(symbol)
        # P/B filter: negative/zero excluded (negative equity is a
        # distress signal, not a valid peer comparison point); very
        # high P/B (>50) usually reflects data errors or extreme
        # asset-light outliers rather than a meaningful peer level.
        if pb and 0 < pb < 50:
            pb_list.append(pb)
            peers_used.add(symbol)

    median_pe = round(float(np.median(pe_list)), 2) if pe_list else None
    median_pb = round(float(np.median(pb_list)), 2) if pb_list else None

    return median_pe, median_pb, sorted(peers_used)


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


def update_valuation_benchmarks():
    """
    Main function: build valuation_benchmarks.json from peer_groups.json
    (FMP-sourced peer identity, refreshed separately and infrequently)
    + live yfinance P/E-P/B + Damodaran fallback.

    P/E has three tiers: FMP peer group -> Damodaran industry -> global default.
    P/B has two tiers only: FMP peer group -> global default (no industry
    fallback dataset exists for P/B).
    """
    print("EquityMind — Valuation Benchmark Updater")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 50)

    peer_groups = _load_peer_groups()
    print(f"\nUniverse size: {len(peer_groups)} tickers")

    damodaran_data = fetch_damodaran_pe()

    valuation_benchmarks = {}
    # Global fallback values, sourced from real S&P 500 aggregate data
    # (GuruFocus / S&P Dow Jones Indices) rather than an arbitrary guess.
    # Update these figures periodically by checking
    # https://www.gurufocus.com/economic_indicators/57/sp-500-pe-ratio
    # and https://www.gurufocus.com/economic_indicators/4240/sp-500-price-to-book-value
    default_pe = 25.54   # S&P 500 P/E, GuruFocus, as of 2026-07-06
    default_pb = 5.44    # S&P 500 P/B, GuruFocus/S&P Dow Jones Indices, as of 2026-01-23
    default_ps = 3.70    # S&P 500 P/S, GuruFocus/S&P Dow Jones Indices, as of 2026-06-15

    for i, (ticker, peer_symbols) in enumerate(peer_groups.items()):
        peer_pe, peer_pb, peers_used = compute_peer_median_pe_pb(peer_symbols, ticker)

        # ── P/E fallback chain: FMP -> Damodaran -> default ──
        if peer_pe:
            pe_source = "fmp_peer_group"
            pe = peer_pe
        else:
            damodaran_pe = get_damodaran_fallback(ticker, damodaran_data)
            if damodaran_pe:
                pe_source = "damodaran"
                pe = damodaran_pe
            else:
                pe_source = "sp500_median"
                pe = default_pe

        # ── P/B fallback chain: FMP -> default (no industry dataset available) ──
        if peer_pb:
            pb_source = "fmp_peer_group"
            pb = peer_pb
        else:
            pb_source = "sp500_median"
            pb = default_pb

        valuation_benchmarks[ticker] = {
            "benchmark_pe": pe,
            "pe_source":    pe_source,
            "benchmark_pb": pb,
            "pb_source":    pb_source,
            "peers_used":   peers_used,
        }

        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(peer_groups)}")

    output = {
        "updated_at": str(date.today()),
        "source":     "yfinance (P/E-P/B) + peer_groups.json (peer identity) + Damodaran industry dataset (P/E fallback)",
        "notes":      "No FMP calls in this script — safe to run daily. Peer identity itself comes from peer_groups.json (see update_peer_groups.py, run separately/infrequently).",
        "default_pe": default_pe,
        "default_pb": default_pb,
        "default_ps": default_ps,
        "benchmarks": valuation_benchmarks,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n✓ valuation_benchmarks.json updated — {len(valuation_benchmarks)} tickers")
    fmp_pe_count = sum(1 for v in valuation_benchmarks.values() if v["pe_source"] == "fmp_peer_group")
    dam_pe_count = sum(1 for v in valuation_benchmarks.values() if v["pe_source"] == "damodaran")
    def_pe_count = sum(1 for v in valuation_benchmarks.values() if v["pe_source"] == "sp500_median")
    fmp_pb_count = sum(1 for v in valuation_benchmarks.values() if v["pb_source"] == "fmp_peer_group")
    def_pb_count = sum(1 for v in valuation_benchmarks.values() if v["pb_source"] == "sp500_median")
    print(f"  P/E — FMP peer group: {fmp_pe_count} | Damodaran: {dam_pe_count} | Default: {def_pe_count}")
    print(f"  P/B — FMP peer group: {fmp_pb_count} | Default: {def_pb_count}")


if __name__ == "__main__":
    update_valuation_benchmarks()
