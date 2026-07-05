"""
scripts/update_peer_groups.py

Peer Group Benchmark Updater for EquityMind Layer 2 Quant Engine.

Data sources:
    Primary:  Financial Modeling Prep API — /stable/stock-peers
              Fetches real peer companies per ticker, computes median P/E
    Fallback: Damodaran industry dataset (NYU Stern, updated annually)
              Used when FMP peer group is unavailable or empty

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
# Companies to build peer benchmarks for
# Add more tickers here as needed
# ─────────────────────────────────────────────
TARGET_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    "AVGO", "ORCL", "CRM", "AMD", "INTU", "IBM", "NOW",
    "NFLX", "DIS", "UBER", "RIVN", "SPCX",
    "JPM", "BAC", "GS", "JNJ", "LLY", "UNH",
    "XOM", "CVX", "WMT", "COST", "PG", "KO",
]

# ─────────────────────────────────────────────
# Damodaran industry P/E mapping
# Maps yfinance industry string → Damodaran industry name
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


def fetch_peer_median_pe(ticker: str) -> float | None:
    """
    Fetch peer group from FMP, compute median P/E of peers.
    Returns None if peer group is empty or all P/E values are missing.
    """
    url = f"https://financialmodelingprep.com/stable/stock-peers?symbol={ticker}&apikey={FMP_API_KEY}"
    r   = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None

    peers = json.loads(r.text)
    if not peers:
        return None

    pe_list = []
    for peer in peers:
        symbol = peer.get("symbol")
        if not symbol or symbol == ticker:
            continue
        try:
            pe = yf.Ticker(symbol).info.get("trailingPE")
            # Filter out extreme values (loss-making or bubble stocks)
            if pe and 3 < pe < 150:
                pe_list.append(pe)
        except Exception:
            continue

    if not pe_list:
        return None

    return round(float(np.median(pe_list)), 2)


def get_damodaran_fallback(ticker: str, damodaran_data: dict) -> float | None:
    """
    Look up Damodaran industry P/E for a ticker using yfinance industry field.
    Returns None if no mapping found.
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
    """
    print("EquityMind — Peer Group Benchmark Updater")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 50)

    # Load Damodaran data once
    damodaran_data = fetch_damodaran_pe()

    peer_benchmarks = {}
    default_pe      = 30  # global fallback

    for ticker in TARGET_TICKERS:
        print(f"\n[{ticker}]")

        # Primary: FMP peer group median P/E
        peer_pe = fetch_peer_median_pe(ticker)
        if peer_pe:
            source = "fmp_peer_group"
            pe     = peer_pe
            print(f"  FMP peer median P/E: {pe}")
        else:
            # Fallback: Damodaran industry P/E
            damodaran_pe = get_damodaran_fallback(ticker, damodaran_data)
            if damodaran_pe:
                source = "damodaran"
                pe     = damodaran_pe
                print(f"  Damodaran fallback P/E: {pe}")
            else:
                source = "default"
                pe     = default_pe
                print(f"  Default fallback P/E: {pe}")

        peer_benchmarks[ticker] = {
            "benchmark_pe": pe,
            "source":       source,
        }

    output = {
        "updated_at": str(date.today()),
        "source":     "FMP stock-peers API + Damodaran industry dataset",
        "notes":      "Run quarterly: python scripts/update_peer_groups.py",
        "default_pe": default_pe,
        "benchmarks": peer_benchmarks,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n✓ peer_benchmarks.json updated — {len(peer_benchmarks)} tickers")
    fmp_count = sum(1 for v in peer_benchmarks.values() if v["source"] == "fmp_peer_group")
    dam_count = sum(1 for v in peer_benchmarks.values() if v["source"] == "damodaran")
    def_count = sum(1 for v in peer_benchmarks.values() if v["source"] == "default")
    print(f"  FMP peer group: {fmp_count} | Damodaran fallback: {dam_count} | Default: {def_count}")


if __name__ == "__main__":
    update_peer_benchmarks()