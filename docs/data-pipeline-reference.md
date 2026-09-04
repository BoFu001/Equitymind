# EquityMind Data Pipeline Reference

Last updated: 2026-08-17

1. [Database tables](#1-database-tables)
2. [Script execution order and dependencies](#2-script-execution-order-and-dependencies) — read this one first
3. [Kestra automation configuration](#3-kestra-automation-configuration)
4. [yfinance and FMP rate limiting](#4-yfinance-and-fmp-rate-limiting)
5. [Log debugging](#5-log-debugging)
6. [How to change the configuration](#6-how-to-change-the-configuration)
7. [Known issues](#7-known-issues)

---

## 1. Database tables

| Table | Purpose | Update frequency | Update method |
|---|---|---|---|
| `stock_universe` | 250-ticker list, market cap, company name, common name, 10-K business overview, industry tags, peer group | Daily (changed 2026-08-10 → 2026-08-14) | Kestra |
| `financial_history` | 33 financial metrics, annual and quarterly | Daily | Kestra |
| `momentum_benchmarks` | Momentum indicators, 52-week high/low position | Daily | Kestra |
| `quant_signals` | Six aggregated signals (valuation, momentum, risk, quality, consensus, short) | Daily | Kestra |
| `sec_chunks` | 10-K filing chunks and embeddings, for pgvector retrieval | On demand | Auto-refreshed at query time |

### `stock_universe` schema, as rebuilt 2026-08-17

Nine columns, in this order. The order is cosmetic — nothing reads by
position — but `scripts/init_db_stock_universe.py` and the live table are kept
identical so a rebuild from that script produces the table now in production.

```
ticker                TEXT PRIMARY KEY
market_cap            BIGINT   -- build_stock_universe.py
company_name          TEXT     -- build_stock_universe.py
common_name           TEXT     -- update_common_names.py
overview_text         TEXT     -- update_stock_overviews.py
overview_filing_date  TEXT     -- update_stock_overviews.py
llm_tags              JSONB    -- update_industry_tags.py
tags_filing_date      TEXT     -- update_industry_tags.py
peers                 JSONB    -- update_peer_groups.py
```

Two columns present before 2026-08-17 are gone: `overview_embedding` (below)
and `updated_at`, which only `update_peer_groups.py` ever wrote and nothing
ever read — its UPDATE statement was amended in the same change. A script
failing with `column "updated_at" does not exist` is hitting this.

### `overview_embedding` dropped, industry tags added (2026-08-17)

**What changed.** `stock_universe.overview_embedding` (a `vector(1536)` of
each company's business overview) was dropped, and
`update_stock_overviews.py` no longer generates it. Two new columns,
`llm_tags` and `tags_filing_date`, are written by a new script,
`update_industry_tags.py`.

**Why.** Discovery used to resolve an open-ended theme ("robotics",
"banking") by embedding the user's phrase and ranking all 250 companies by
cosine similarity against that column. Tag-based retrieval replaced that
mechanism — the LLM picks one tag from a fixed vocabulary and the exact
ticker list for that tag is returned — which left the embedding column with
no consumers at all. A grep across the codebase found four references, all of
them schema definitions or the write path itself; nothing read it.

Continuous similarity ranking has no natural stopping point: for a narrow
query it admitted large amounts of category-irrelevant noise, and no single
threshold worked across both broad and narrow queries. Exact tag lookup is
bounded by construction, and can return nothing at all, which the similarity
ranking could not express.

**The frozen baseline.** The 243×1536 matrix, together with the
`overview_text` that produced each vector, the embedding model name and the
date, was exported before the column was dropped and is kept outside the
repository. **It is the only copy.** It is needed because specific similarity
scores from that matrix are cited in the research logs, and neither the
embedding model's future behaviour nor the overviews themselves (which are
regenerated as new 10-Ks land) can be relied on to reproduce them.

### Industry tags: what they are and how they stay fresh

`llm_tags` is a JSONB array of industry/theme tags for one company —
`["biotechnology", "pharmaceuticals", "veterinary medicine", "diagnostics"]`
for ZTS, typically three to five entries. They are extracted from that
company's `overview_text`, not from the 10-K directly: the overview generation
step already writes a closing classification sentence, and the extraction step
restates the tags named in it.

`tags_filing_date` records which 10-K's overview a company's tags came from. A
ticker is re-tagged only when it diverges from `overview_filing_date`:

```
if tags_filing_date != overview_filing_date:
    re-extract, then set tags_filing_date = overview_filing_date
```

Deciding freshness from the database rather than from EDGAR is deliberate, and
matters in exactly the case that goes wrong. Suppose a new 10-K lands,
`update_stock_overviews.py` succeeds and advances `overview_filing_date`, and
then `update_industry_tags.py` fails — OpenAI credit exhausted, a timeout, the
process killed. A script that asked EDGAR on the next run would find the
filing date it already knows about, conclude nothing is new, and skip the
ticker; the stale tags would survive indefinitely with no signal that anything
was wrong. Comparing the two stored dates leaves the divergence visible in the
data, so the next run picks the ticker up on its own. This is the same
self-healing principle as the `peers` NULL fix below — do not let a failure
look like a success.

The side effect is that the script is free to run daily. With no new 10-K it
is one query and exits in under a second; a production run on 2026-08-17 took
0.8s end to end.

Tickers with a NULL `overview_text` (C, DVN, FANG, GE, KMI, PSX, SPCX — see
`update_stock_overviews.py`'s docstring) are skipped and keep NULL tags, which
makes them invisible to every tag-based query with no special-case code.

### Tag normalisation happens at query time, not in the database

Each 10-K is summarised independently, so the same concept surfaces under
variant spellings depending on how a given filing phrases it. Two real splits
in the current data: `semiconductors` (11 companies) against `semiconductor`
(4), and `ai` (10) against `artificial intelligence` (9, overlapping only at
SNOW). Because the LLM picks exactly one tag per query, an unmerged split
silently halves the answer — "AI companies" returned the nine
`artificial intelligence` holders and omitted NVDA, MSFT and GOOGL, with no
error and nothing in the output to suggest anything was missing.

`llm_tags` stores the raw extraction. Normalisation lives in
`get_industry_tickers`, which rebuilds a `{tag: [tickers]}` index per call:

- **Plurals are a rule**: strip a trailing `s`, and merge only if the singular
  already exists as a tag in its own right. Industry nouns pluralise plainly
  (semiconductors, devices, pharmaceuticals) so one rule covers them, and the
  existence check leaves `logistics`, `diagnostics` and `analytics` alone. The
  worst case is a missed merge, never a wrong one.
- **Everything else goes in `TAG_SYNONYMS`**, a small dict for variants no rule
  can express (`ai` → `artificial intelligence`).

Normalising at query time rather than writing normalised tags back means a
rule change takes effect on the next query with no data migration, and the
wording each filing actually used is never overwritten — which is how both
splits above were found in the first place. Current effect: 271 raw tags → 263
after normalisation.

Rebuilding the index per call is deliberate: it is one SELECT over ~250 rows
sitting next to an LLM call that costs three orders of magnitude more, in a
node that already queries the same table. Caching would buy nothing measurable
and would add a staleness mode.

### `stock_universe` moved to daily automation (2026-08-14)

Previously `stock_universe` was rebuilt "every few months, manually" — the
four scripts that write it were run by hand.

Why this changed: `stock_universe.market_cap` needed to be current for
Discovery's market-cap-based ranking queries ("the 10 largest companies by
market cap"). A market cap that is months stale produces wrong rankings.
Rather than build a separate, narrower "just refresh market_cap" script, the
existing `build_stock_universe.py` was moved to run daily as-is — it already
re-ranks the full candidate pool and updates `market_cap` / `company_name` for
the current top 250, so daily execution keeps market cap current as a side
effect of its normal behaviour.

The other three scripts were made incremental so that daily execution doesn't
waste API calls re-processing tickers that are already up to date:

- `update_common_names.py` — now only processes tickers where
  `common_name IS NULL` (previously re-ran the LLM extraction for all 250
  every time).
- `update_peer_groups.py` — now only processes tickers where `peers IS NULL`.
  See the NULL-vs-empty-list fix below — this is the field most likely to have
  a stale or incorrect empty result if you're debugging Discovery.
- `update_stock_overviews.py` — was already incremental (compares each
  ticker's latest 10-K filing date against `overview_filing_date`).

`update_industry_tags.py`, added 2026-08-17, follows the same incremental
pattern from the start.

### `peers` — empty result now stored as NULL, not `[]` (fixed 2026-08-14)

**Bug found:** `update_peer_groups.py` previously wrote `[]` to
`stock_universe.peers` whenever `fetch_peer_symbols()` returned no usable
peers — whether because FMP genuinely had no peer data (e.g. SPCX, a pre-IPO
company), or because the FMP request itself failed (timeout, non-200 status,
exception). Both cases produced the same `[]` result, which is
indistinguishable from "already successfully processed, no peers exist."
Because `_load_target_tickers()`'s incremental filter is `WHERE peers IS NULL`,
a ticker written with `[]` would never be revisited — a transient FMP failure
became a permanent incorrect result with no automatic recovery path.

**Confirmed case:** HAL (Halliburton) was written as `[]` on 2026-08-14. A
manual re-fetch minutes later, using the exact same `fetch_peer_symbols()`
call, returned 10 real peers (8 after USD-reporter filtering: CQP, DVN, EXE,
EXEEZ, FTI, TPL, TS, VG). This confirms the empty result was a transient
failure, not a genuine "no peers" case.

**Fix:** `update_peer_groups.py` now leaves `peers` as NULL — does not write to
the database at all — whenever the final filtered peer list is empty,
regardless of whether that's from a fetch failure, a genuine empty FMP
response, or every candidate peer being filtered out for non-USD reporting.
The three cases are deliberately not distinguished, so the ticker is
automatically retried on the next daily run.

**Trade-off, accepted deliberately:** tickers with genuinely no peers (e.g.
pre-IPO companies like SPCX) will be re-queried against FMP every day, costing
one wasted API call per day indefinitely. This is accepted in exchange for
self-healing — the day FMP does add peer data for such a ticker, it is picked
up automatically with no manual intervention. FMP's 250/day quota comfortably
absorbs this; see section 4.

**Diagnosing this class of issue:** `fetch_peer_symbols()` now logs the
specific failure reason (`FMP returned status {code}` or
`FMP request failed: {exception}`) rather than silently returning `[]`. Search
Railway logs for these strings when investigating why a ticker isn't getting
peer data.

---

## 2. Script execution order and dependencies

All eight scripts run daily, in one continuous Kestra flow. There is no longer
a separate "low-frequency, manual" tier.

**This section is the one to re-read first after time away from the project.**
The running order below is what the flow does; the part worth actually
remembering is which parts of that order are load-bearing and which are
arbitrary.

### 2.1 What each script reads and writes

Everything writes to `stock_universe` except the last three, which write their
own tables. "Reads" means a dependency — a column that must already be
populated for the script to do useful work.

| Script | Reads | Writes | Incremental filter |
|---|---|---|---|
| `build_stock_universe.py` | nothing (external: yfinance) | `ticker`, `market_cap`, `company_name` | none; re-ranks the full candidate pool every run |
| `update_common_names.py` | `company_name` | `common_name` | `WHERE common_name IS NULL` |
| `update_peer_groups.py` | `ticker` | `peers` | `WHERE peers IS NULL` |
| `update_stock_overviews.py` | `ticker`, `overview_filing_date` | `overview_text`, `overview_filing_date` | EDGAR filing date vs. `overview_filing_date` |
| `update_industry_tags.py` | `overview_text`, `overview_filing_date` | `llm_tags`, `tags_filing_date` | `tags_filing_date IS DISTINCT FROM overview_filing_date` |
| `update_momentum_benchmarks.py` | `ticker` | `momentum_benchmarks` table | none; recomputes all |
| `update_financial_history.py` | `ticker` | `financial_history` table | per-ticker freshness check |
| `update_quant_signals.py` | `ticker`, `peers`, `financial_history`, `momentum_benchmarks` | `quant_signals` table | none; recomputes all |

The whole graph reduces to one rule and three edges: every script needs the
ticker list, so `build_stock_universe.py` goes first; after that, only three
pairs actually constrain each other.

### 2.2 The three hard dependencies

1. **`build_stock_universe.py` before everything.** It decides who is in the
   universe. The other seven only ever process tickers already on that list,
   so a ticker added today gets its name, peers, overview, tags and financial
   data on the same run — not a day later.
2. **`update_stock_overviews.py` before `update_industry_tags.py`.** Tags are
   extracted from `overview_text`, not from the 10-K. Reversed, a company
   whose overview was refreshed today would be tagged from yesterday's text.
3. **`update_momentum_benchmarks.py` and `update_financial_history.py` before
   `update_quant_signals.py`.** Quant signals are computed from those two
   tables — `financial_history` feeds the Quality signal,
   `momentum_benchmarks` the Momentum signal. Running quant signals first
   computes them from the previous day's inputs.

### 2.3 What is not a dependency

`update_common_names.py`, `update_peer_groups.py` and
`update_stock_overviews.py` are mutually independent. They read different
columns, write different columns, and call different APIs. Their current
relative order is historical, not required — reordering them, or running one
alone to debug it, is safe.

Likewise, the tag chain (overviews → tags) and the quant chain (momentum +
financial history → quant signals) do not touch each other at all. Stage 2
runs after Stage 1 only because both need a current ticker list, not because
quant signals care about tags or overviews.

This matters because it tells you what you can safely change. If a step needs
to move — because it is slow, or failing, or you want it on a different
schedule — check it against the three dependencies above. Anything not listed
there can be moved freely.

### 2.4 Getting the order wrong: two very different outcomes

Not all ordering mistakes are equal, and the difference decides whether one
needs fixing.

**Self-healing (costs a day, not correctness).** Tags falling behind
overviews. Because `update_industry_tags.py` compares `tags_filing_date`
against `overview_filing_date` rather than asking EDGAR, a company that missed
its re-tag leaves the two dates divergent, and the next run picks it up with
no intervention. The same applies to the whole script failing — credit
exhausted, timeout, killed process. Nothing to repair.

**Not self-healing (silently wrong data).** Quant signals computed from an
incomplete `financial_history`. `update_quant_signals.py` recomputes every
ticker on every run and keeps no record of which inputs it used, so a Quality
score derived from a half-written `financial_history` looks exactly like a
correct one and stays wrong until the next daily run overwrites it. This is why the
Stage 2 sequence is enforced strictly rather than by convention.

The general shape: **a step that records what its output was derived from can
recover on its own; a step that just overwrites cannot.** Worth keeping in
mind when adding anything new to the flow.

### 2.5 The running order

**Stage 1 — `stock_universe`.** Must complete before Stage 2, because
`financial_history` and `quant_signals` both read the current universe list:

```
build_stock_universe.py           [hard: must be first]
    ↓ (wait for success)
update_common_names.py       ┐
    ↓                        │  mutually independent —
update_peer_groups.py        │  order among these three
    ↓                        │  is historical, not required
update_stock_overviews.py    ┘
    ↓
update_industry_tags.py           [hard: must follow update_stock_overviews.py]
```

**Stage 2 — market data and computed signals:**

```
update_momentum_benchmarks.py ┐
    ↓                         │  both must precede quant_signals;
update_financial_history.py   ┘  order between these two is free
    ↓
update_quant_signals.py           [hard: must be last]
```

Each arrow is a Kestra `LoopUntil` that polls until the previous script
reports success, so no two scripts overlap (see section 3).

### 2.6 Manual run commands

Any script, any time — useful for debugging a single step without running the
whole flow:

```bash
python scripts/build_stock_universe.py
python scripts/update_common_names.py
python scripts/update_peer_groups.py
python scripts/update_stock_overviews.py
python scripts/update_industry_tags.py
python scripts/update_momentum_benchmarks.py
python scripts/update_financial_history.py
python scripts/update_quant_signals.py
```

Because of the incremental filters in 2.1, running any of these out of order
is safe — a script with nothing to do exits immediately without calling any
API. The exception is `update_quant_signals.py`, which always recomputes:
running it on its own is the correct fix after a partial failure, but running
it before its two inputs on the same day gives stale signals with no warning.

Two flags worth knowing:

- `update_stock_overviews.py --all` regenerates every overview, ignoring
  filing dates. Expensive — 243 heavier-model calls over ~50k characters of
  Item 1 text each.
- `update_industry_tags.py --all` re-tags every ticker, ignoring
  `tags_filing_date`. Use after changing the extraction prompt. Costs one
  light-model call per ticker and, unlike clearing `overview_filing_date` to
  force a re-tag, does not regenerate the overviews to get there.

---

## 3. Kestra automation configuration

- **Service URL:** `https://kestra-production-6b60.up.railway.app`
- **Railway project:** `equitymind-pipeline` (independent from `equitymind-core`)
- **Flow:** `equitymind_daily_pipeline` (namespace `equitymind`)
- **Trigger:** daily, 04:00 UK local time (`cron: "0 4 * * *"`, `timezone: "Europe/London"`)

The flow definition is kept in this repository at
`docs/kestra/equitymind_daily_pipeline.yml`. That file is a copy for reference
and version control — **editing it changes nothing**; the flow that actually
runs lives in Kestra and must be edited there.

**Trigger mechanism** — Kestra calls `equitymind-core`'s internal HTTP
endpoints:

- `POST /api/v1/internal/run-pipeline/{script_name}` (trigger)
- `GET /api/v1/internal/run-pipeline/{run_id}` (check status)

Authentication: header `x-pipeline-secret`, value from
`PIPELINE_TRIGGER_SECRET` (must be set identically on both `equitymind-core`
and Kestra).

**Dependency enforcement:** a `LoopUntil` task polls the `run_id` status every
10 seconds until status becomes `"success"`, before triggering the next
script. The flow has eight trigger/wait pairs — Stage 1's five steps precede
the original three.

### Timeout values (as of 2026-08-17)

| Step | maxIterations | Effective timeout | Rationale |
|---|---|---|---|
| `wait_for_stock_universe` | 180 | 30 min | Throttles its yfinance calls across the full candidate pool (400+ tickers) — see section 4. |
| `wait_for_common_names` | 60 | 10 min | Incremental — most days process 0 or a handful of tickers. A run still going after 10 min indicates a real problem. |
| `wait_for_peer_groups` | 60 | 10 min | Same reasoning as common_names. |
| `wait_for_stock_overviews` | 180 | 30 min | Already incremental, but a run regenerating several overviews (one heavier-model call each, on ~50k characters) can take longer. |
| `wait_for_industry_tags` | 120 | 20 min | Added 2026-08-17. Normally a no-op finishing in under a second. The timeout only needs to cover the rare full re-tag, which took roughly 15 min for 243 tickers. |
| `wait_for_momentum_benchmarks` | 90 | 15 min | Unchanged from 2026-08-10. |
| `wait_for_financial_history` | 180 | 30 min | Unchanged. Actual run time ~13 min on Railway (throttle=2s). |
| `wait_for_quant_signals` | 180 | 30 min | Unchanged. Actual run time ~17 min under normal conditions — see the known issue below for a run that a transient yfinance rate limit pushed to 63/250 succeeded. |

`update_industry_tags.py` calls no rate-limited third-party data API — only
OpenAI, and only for tickers whose overview actually changed — so it adds
nothing to the yfinance or FMP budgets in section 4.


### Kestra memory cap (set 2026-09-04)

Kestra only schedules the eight scripts; the heavy work runs in the
script processes it launches. Without a cap the JVM grows toward the
machine's memory share (measured ~2.5 GB resident, ~$25/month on
Railway). Set on the Railway Kestra service:

    JAVA_OPTS=-Xms256m -Xmx512m

Scheduling is unaffected. If Kestra ever hits the cap and restarts
overnight, the incremental filters pick up any missed rows the next
night (the usual self-healing path).


---

## 4. yfinance and FMP rate limiting

### yfinance

Used by `build_stock_universe`, `financial_history`, `momentum_benchmarks` and
`quant_signals`.

**Problem, historical (documented 2026-08-10):** running the batch scripts on
Railway triggers yfinance rate limiting; this does not happen when run
locally. Measured throttle values and outcomes:

| Scenario | Throttle | Result |
|---|---|---|
| Local run | 0 | 250/250 succeeded |
| Railway, no concurrent queries | 0.5s | Large-scale failure |
| Railway, no concurrent queries | 0.1s | Worse |
| Railway, no concurrent queries | 1s | Started failing around ticker #111 |
| Railway, no concurrent queries | 2s | 250/250 succeeded (correct) |
| Railway, with concurrent user query | 2s | 172/250 succeeded (insufficient) |

**Current setting:** `PIPELINE_THROTTLE_SECONDS=2`, set in `equitymind-core`'s
Railway environment variables (not Kestra). The throttle sleep fires once per
ticker in each script's main loop, after that ticker's API calls are done —
not once per individual API call.

`build_stock_universe.py` uses this same throttle as of 2026-08-14.
Previously it had none, iterating the full candidate pool (400+ tickers)
calling `yf.Ticker(symbol).info` with no delay. Low-risk at a few runs per
year, a real concern at one run per day.

**New observation (2026-08-14):** even at throttle=2s, yfinance rate limiting
can still occur, transiently. On one daily run, `update_quant_signals.py`
failed for 187/250 tickers with repeated
`Too Many Requests. Rate limited. Try after a while.` errors in a sustained
window (~17:05–17:11). A re-trigger of the same script shortly afterward
completed 250/250 with zero errors — same code, same throttle. This suggests
cumulative request volume across the day's earlier steps landing during a
Yahoo Finance rate-limit window, not a flaw in the throttle logic.
`update_quant_signals.py` recomputes all 250 tickers on every run, so a full
re-trigger is sufficient recovery. If this recurs frequently, consider a
longer throttle or a retry-with-backoff wrapper around the yfinance calls in
`quant_signals` specifically.

### FMP (`update_peer_groups.py`)

**Quota:** 250 requests/day, shared across the whole universe. The incremental
filter (`WHERE peers IS NULL`) is what makes daily execution safe within this
quota — on a typical day with 0–3 tickers newly missing peer data, the script
uses a small fraction of it. Before this script was made incremental, running
it daily against the full universe would have used the entire quota every day.

At 04:00 UK time there is negligible concurrent load, so FMP rate limits have
not been observed. Note the NULL/`[]` fix above means occasional single-ticker
FMP failures are now expected and self-correcting, not something to chase down
individually.

### OpenAI

Used by `update_common_names.py`, `update_stock_overviews.py` and
`update_industry_tags.py`. No rate limit observed. All three are incremental,
so on a typical day with no new 10-K they make zero API calls between them.
The cost concentrates in the rare full rebuild: 243 heavier-model calls for
overviews, 243 light-model calls for tags.

---

## 5. Log debugging

On Railway, in `equitymind-core`'s Deploy Logs, search for `[PIPELINE]` to
filter out the full output of all batch scripts (including per-ticker
progress), without being buried in concurrent user-query logs.

All print statements across all scripts use `flush=True` (added 2026-08-14).
Previously, `build_stock_universe.py`, `update_common_names.py`,
`update_peer_groups.py`, and most of `update_stock_overviews.py` used Python's
default block buffering under `subprocess.PIPE` — output was held until the
buffer filled or the process exited, so Railway logs showed only HTTP polling
records with no visible script progress until the very end of a run, or
sometimes not at all.

Each script prints a final summary:

- **`update_financial_history.py`** — `✓ financial_history updated — {N} rows
  written across {M} tickers`, plus `Annual — failed: [...]` /
  `Quarterly — failed: [...]` only if non-empty.
- **`update_quant_signals.py`** — `✓ quant_signals updated — {N}/{M} tickers
  succeeded`, plus `Failed: [...]` only if non-empty.
- **`update_momentum_benchmarks.py`** — `✓ momentum_benchmarks table updated —
  {N}/{M} tickers written`.
- **`build_stock_universe.py`** — `✓ stock_universe table updated — {N}
  tickers`, plus `Departed: [...]` / `Arrived: [...]` listing which tickers
  left or entered the top 250.
- **`update_common_names.py`** — `Tickers needing a common_name: {N}` at the
  start, `✓ stock_universe table updated — {N} new common_name(s) written` at
  the end.
- **`update_peer_groups.py`** — `Tickers needing peers: {N}` at the start;
  `✓ stock_universe table updated — {N}/{M} new tickers now have an FMP peer
  group` at the end, plus `{N} ticker(s) left NULL (empty result) — will retry
  next run` if any.
- **`update_stock_overviews.py`** — `✓ stock_universe table updated — {N}
  overview(s) written`, plus `No usable Item 1 for {N}: [...]` (known cases: C,
  DVN, FANG, GE, KMI, PSX, SPCX).
- **`update_industry_tags.py`** — `{N} ticker(s) need tag extraction` at the
  start, the extracted tag list per ticker, then `✓ stock_universe table
  updated — {N} ticker(s) tagged`, plus `Extraction failed for {N}: [...]` if
  any response could not be parsed. **On a normal day the expected output is
  `0 ticker(s) need tag extraction`** — a non-zero count means a 10-K landed,
  and a count near 243 means something cleared `tags_filing_date` wholesale.

Each `Run <uuid> (scripts/xxx.py) finished with status: success` line marks the
true end of one script's execution — searching for these across a time range is
the most reliable way to confirm how many times a script actually ran and
whether two runs overlapped.

---

## 6. How to change the configuration

**Change trigger time.** In Kestra, edit `equitymind_daily_pipeline`, change
the cron expression.

**Change throttle value.** In Railway, `equitymind-core` service → Variables,
edit `PIPELINE_THROTTLE_SECONDS`, save (triggers automatic redeploy). Affects
`build_stock_universe.py` in addition to the three Stage 2 scripts.

**Change the secret.** `PIPELINE_TRIGGER_SECRET` must be updated on both
`equitymind-core` and Kestra simultaneously, otherwise requests fail with 401.

**Change Kestra timeouts.** Edit `checkFrequency.maxIterations` on the
relevant `LoopUntil` task in the flow YAML (section 3). Eight tasks.

**Register a new pipeline script.** Two edits, both required:

1. Add it to `ALLOWED_SCRIPTS` in `api/routes/pipeline.py` — a whitelist; the
   endpoint rejects any script name not explicitly listed.
2. Add the corresponding `trigger_*` / `wait_for_*` task pair to the Kestra
   flow.

Adding to one without the other either makes the script untriggerable (missing
from the allowlist) or leaves it registered but never called (missing from the
flow). The whitelist is written in execution order for readability only —
ordering there has no effect, since it is a lookup dict and Kestra owns
sequencing.

**Deploy order matters:** push the script and the whitelist entry first, wait
for Railway to redeploy, then add the Kestra tasks. The other way round gives
a 404 on the first run.

**Add a tag synonym.** Edit `TAG_SYNONYMS` alongside `get_industry_tickers`.
Plural variants need no entry — the rule handles them. Takes effect on the
next query with no re-tagging and no database change. Adding an entry that
maps a tag onto one that does not exist in the data silently creates an empty
tag, so check the target against the current vocabulary first.

---

## 7. Known issues

### Transient yfinance rate limiting during quant_signals

*Observed 2026-08-14, not yet a recurring pattern.*

See section 4. One run failed 187/250 tickers on sustained "Too Many Requests"
errors; an isolated re-trigger succeeded 250/250 with no code changes. Not yet
confirmed whether this is a one-off or a pattern. If it recurs on multiple
days, investigate a longer throttle or a retry-with-backoff wrapper
specifically around `quant_signals`'s yfinance calls — `update_financial_history.py`
already throttles per-ticker but this failure occurred in `quant_signals`,
which has its own separate call pattern (`get_stock_snapshot`,
`get_valuation_inputs`, `get_momentum_inputs`, `get_risk_inputs`, etc. per
ticker).
