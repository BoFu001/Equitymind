"""
scripts/init_db_quant_signals.py

One-time database setup script for EquityMind's precomputed quant
signal store — creates the quant_signals table in the same PostgreSQL
database used for sec_chunks and financial_history.

Design: each signal's COMPLETE return dict (from valuation_signal(),
momentum_signal(), risk_signal(), quality_signal(), consensus_signal(),
short_signal()) is stored verbatim in a JSONB column — not hand-picked into separate
flat columns. This guarantees the cached path and the live-computation
path are always identical: format_valuation(), format_risk(), etc. can
be called on either json.loads(row["valuation_data"]) or a live
valuation_signal() result with zero special-casing, and adding a new
field to any signal function requires no table migration (it just
appears in the JSONB automatically) — this avoids the class of
silent-drift bug already seen once in this project (company_name
handling in news_reader.py).

A small number of scalar columns are duplicated out of each JSONB
column specifically to support Discovery's screening queries (e.g.
"undervalued quality stocks", "high risk/high reward") — these need
native SQL WHERE/ORDER BY, which JSONB alone does not support
efficiently. Some are derived scores, some are the raw figure itself
(consensus_recommendation_mean is the 1-5 analyst mean, so it sorts
ascending); the column name always says which. These are the ONLY
columns that must be remembered when a signal's fields change;
everything else is automatic via the JSONB column.

Column names for these scalar columns match the corresponding key
inside the signal's JSONB exactly (e.g. position_52w_score, not
momentum_52w_score) — deliberately not prefixed with the signal group
name, so there is never any ambiguity about which JSONB key a scalar
column was copied from. See momentum_signal.py: the 52-week-high
sub-signal's field is named "position_52w_score" internally (it
measures nearness to the 52-week high, not "momentum" in the 12-1
sense), and the table column matches this verbatim.

News Sentiment is deliberately excluded — never cached, always fetched
and scored live (see news_reader.py / news_sentiment_signal.py), due to
its short shelf life.

Scope: only the 250-ticker universe is stored here. Tickers outside the
universe are computed live and NOT written to this table — they would
silently rely on a generic (non-peer-specific) valuation benchmark
(see valuation_signal.py's DEFAULT_PE/DEFAULT_PB fallback), which would
pollute screening queries with lower-confidence data if persisted.

All six signals are recomputed on every run, so each *_computed_at
column records when that run happened rather than distinguishing one
signal group's cadence from another's.

quality_period_end: which financial_history period_end the cached
Quality result was computed from (distinct from quality_computed_at,
which only says when). Lets update_quant_signals.py skip recomputing
Quality when financial_history has no newer annual period than this.
Nullable — unlike financial_history's one-row-per-period design, this
table has one row per ticker covering all 6 signals, so a missing
Quality result must not block the row's other signals from being written.

Run once to create the table:
    python scripts/init_db_quant_signals.py
"""


import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_URL



def init():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    print("Creating quant_signals table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quant_signals (
            ticker                          TEXT PRIMARY KEY,

            -- Valuation (see valuation_signal.py for full return shape)
            valuation_data                  JSONB,
            valuation_score                 REAL,
            valuation_computed_at           TIMESTAMP,

            -- Momentum (see momentum_signal.py)
            momentum_data                   JSONB,
            momentum_12_1_score             REAL,
            position_52w_score              REAL,
            momentum_computed_at            TIMESTAMP,

            -- Risk (see risk_signal.py)
            risk_data                       JSONB,
            risk_beta_score                 REAL,
            risk_sharpe_score               REAL,
            risk_var_score                  REAL,
            risk_drawdown_score             REAL,
            risk_computed_at                TIMESTAMP,

            -- Quality (see quality_signal.py)
            quality_data                    JSONB,
            quality_score                   REAL,
            quality_period_end              DATE,
            quality_computed_at             TIMESTAMP,

            -- Consensus (see consensus_signal.py)
            consensus_data                  JSONB,
            consensus_recommendation_mean   REAL,
            consensus_upside_pct            REAL,
            consensus_computed_at           TIMESTAMP,

            -- Short (see short_signal.py)
            short_data                      JSONB,
            short_interest_pct              REAL,
            days_to_cover                   REAL,
            short_computed_at               TIMESTAMP
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()

    print("quant_signals table ready.")


if __name__ == "__main__":
    init()
