# equitymind-core

The backend of EquityMind: an agentic RAG system that answers plain-English
questions about US-listed equities with retrieved filings, computed
quantitative signals, and an explicit account of where each answer came from.

This repository holds everything that decides what an answer contains — the
LangGraph pipeline, the quant engine, the data readers, the SEC ingestion
pipeline, and the HTTP/WebSocket API. It is the only one of the three
repositories that talks to a database or to an LLM.

---

## The three repositories

| Repository | Role |
|---|---|
| `equitymind-core` | This one. Agent graph, quant signals, readers, API, database. |
| `equitymind-web-backend` | Thin proxy. Terminates the browser WebSocket and forwards to core. No business logic, no database. |
| `equitymind-web-frontend` | React + Vite SPA, deployed on Vercel. |

The split exists so the consumer product and any future third-party
integration enter through the same door: `equitymind-web-backend` is simply the
first client of core's API, not a privileged path into it. Authentication is
core's job — `/query/stream` checks the `Authorization` header itself and
closes with code 4001 on failure — which is why the proxy holds no auth logic
of its own.

---

## How a question is answered

Every request enters the graph at `contextualize_question` and leaves at
`update_session_memory`. Between those two, the route depends on what the
question turns out to be.

```
__start__
    ↓
contextualize_question          rewrite the question using session history
    ↓
classify_top_intent             TASK / CONCEPT / GREETING / OUT_OF_SCOPE
    ↓
   ┌─────────────┬──────────────┬─────────────┬──────────────┐
   ↓             ↓              ↓             ↓              ↓
classify_sub_intent      explain_concept  handle_greeting  handle_out_of_scope
   ↓
   ├── DISCOVERY ──→ discovery_preparation ──→ discovery ──┐
   │                          │                            │
   │                  (nothing rankable —                  │
   │                   asks a follow-up,                   │
   │                   turn ends here)                     │
   │                          │                            │
   └── SPECIFIC_STOCK / COMPARISON ──→ extract ────────────┤
                                  │                        │
                          (no ticker found)                │
                                  ↓                        ↓
                            no_ticker           determine_data_scope
                                  │                        ↓
                                  │                   fetch_data
                                  │                        ↓
                                  │                   quant_engine
                                  │                        ↓
                                  │                  generate_report
                                  │                        │
                                  └───────────┬────────────┘
                                              ↓
                                   update_session_memory
                                              ↓
                                           __end__
```

Four things about this shape are worth knowing before changing it.

**`contextualize_question` runs unconditionally**, with no bypass. Its output
feeds intent classification, ticker extraction and scope selection, so a bad
rewrite is a bad rewrite for the whole request.

**`discovery_preparation` is a gate, not a preprocessing step.** Discovery
narrows a pool by ranking it on fields it holds data for; with no rankable
field, every stage is a no-op and the entire pool comes back looking like an
answer. The gate reads the raw `state["question"]`, parses it, and passes it
through only if the parse produced something rankable — otherwise it writes a
clarifying question and the turn ends. It folds conversation history into the
question **only when the previous turn left a Discovery request open**
(`in_clarification`); doing that unconditionally carried an old exchange into
every question after it.

**Both paths converge on `determine_data_scope`**, which decides which of the
nine sources this question needs. It reads
`enriched_query → contextualized_question → question`, in that order — after a
multi-turn Discovery exchange the latest message is a fragment ("the cheapest
ones") and only the enriched question still carries what was asked for. Scope
is decided from the question rather than from Discovery's parsed ranking
fields, because the two answer different things: "companies with major recent
risks" parses to `risk_beta_score`, a price-volatility measure, while what the
question needs is `sec_filing`.

**Every branch converges on `update_session_memory`**, including the greeting
and out-of-scope handlers. Anything written to session state there is written
on every path, not only the analytical ones.

---

## Layout

```
api/            FastAPI application, all routes under /api/v1
  routes/
    query.py        WS   /query/stream         main path, streams tokens
    research.py     POST /query/sync           same graph, one blocking
                                               response. Used by the batch
                                               experiment scripts.
    pipeline.py     POST /internal/run-pipeline/{script_name}
                    GET  /internal/run-pipeline/{run_id}
                                               what Kestra calls. Guarded by
                                               the x-pipeline-secret header
                                               and an ALLOWED_SCRIPTS
                                               whitelist.
    health.py       GET  /health
  auth.py       API key verification
  schemas.py    Request/response models

core/           Cross-layer plumbing. Imports nothing from api/ or src/.
  context.py      ContextVar carrying the per-request token queue, so agent
                  nodes can stream tokens out to a WebSocket handler in
                  another layer without putting live objects in AgentState.

src/
  agent/
    graph.py            Graph construction and routing
    state.py            AgentState (TypedDict)
    capabilities.py     Shared capability list for the three conversational
                        handlers, phrased as copyable example questions
    nodes_notifications.py  Progress strings pushed to the client per node
    discovery_types.py  DiscoveryQuery, RankField, the valid field vocabulary
    nodes/              One file per node. Nodes own progress notifications
                        and state; they call readers and quant functions but
                        contain no retrieval or maths themselves.
    formatters/         Turn retrieved data into the text blocks that go into
                        the report prompt. One per signal or data source.
  quant/          Signal computation. Pure functions over already-fetched
                  data — no database, no LangGraph, no network.
  readers/        Data access. One reader per source/table.
  sec_pipeline/   10-K download, chunking, embedding, pgvector storage

scripts/        Database initialisation and the eight daily pipeline scripts.
                filing_check.py is NOT a script — it is a shared module
                (is_usd_reporter, files_20f_only) that build_stock_universe.py
                and update_peer_groups.py import.
docs/           Pipeline reference and a copy of the Kestra flow definition
tests/          pytest suite, plus a WebSocket debug client
config.py       Model names, environment variables, tuning constants
colors.py       gprint / mprint — coloured console output used by nodes
```

The `agent → readers → quant → formatters` separation is deliberate and load
bearing: **RAG handles fact retrieval, quant handles signal computation, and
the LLM only turns already-computed results into prose.** A node that computes
a ratio, or a quant function that opens a database connection, is a bug even
if it works.

---

## Setup

```bash
conda activate equitymind-core
pip install -r requirements.txt
```

`.env` in the repository root:

| Variable | Purpose | Required |
|---|---|---|
| `OPENAI_API_KEY` | LLM calls and embeddings | yes |
| `DATABASE_URL` | PostgreSQL connection string | yes |
| `FMP_API_KEY` | Peer groups, financial data. 250 requests/day. | yes |
| `FINLIGHT_API_KEY` | News retrieval | yes |
| `PIPELINE_TRIGGER_SECRET` | Shared secret for the Kestra endpoints. Must match the value set in Kestra, or every trigger returns 401. | yes |
| `PIPELINE_THROTTLE_SECONDS` | Per-ticker sleep in the yfinance-calling scripts. Defaults to `0`; production uses `2`. | no |

Models are set in `config.py`, not the environment:

```python
LLM_MODEL       = "gpt-5.2"        # reports, classification, Discovery
LLM_MODEL_LIGHT = "gpt-4o-mini"    # ticker extraction, data scope, news, greeting
```

---

## Database

Five tables. `scripts/init_db_*.py` create the schemas; the `update_*` scripts
populate them.

| Table | Contents | Refreshed |
|---|---|---|
| `stock_universe` | 250 tickers: market cap, names, 10-K business overview, industry tags, peer group | daily |
| `financial_history` | 33 financial metrics, annual and quarterly | daily |
| `momentum_benchmarks` | Momentum indicators, 52-week price position | daily |
| `quant_signals` | Six precomputed signals, each stored whole as JSONB plus scalar columns for screening | daily |
| `sec_chunks` | 10-K chunks and their embeddings, for pgvector retrieval | on demand, at query time |

### Cold start

Create the schemas first, then run the eight pipeline scripts in the order
given in `docs/data-pipeline-reference.md`:

```bash
python scripts/init_db_stock_universe.py
python scripts/init_db_financial_history.py
python scripts/init_db_momentum_benchmarks.py
python scripts/init_db_quant_signals.py
python scripts/init_db_sec_chunks.py

python scripts/build_stock_universe.py
python scripts/update_common_names.py
python scripts/update_peer_groups.py
python scripts/update_stock_overviews.py
python scripts/update_industry_tags.py
python scripts/update_momentum_benchmarks.py
python scripts/update_financial_history.py
python scripts/update_quant_signals.py
```

The `init_db_*` scripts are idempotent — every statement in them is
`CREATE ... IF NOT EXISTS`, so running them against a populated database does
nothing and destroys nothing.

**They are not migration tools.** If a table already exists with an older
schema, the script silently does nothing: no error, no change. Altering a
schema means doing it by hand, as the 2026-08-17 `overview_embedding` removal
did — rename the old table, run the init script to create the new one, copy the
data across, verify, then drop the backup.

A first run is slow and expensive: every overview is generated from ~50k
characters of Item 1 text, every ticker is tagged, and nothing is cached yet.
Subsequent runs are incremental and usually no-ops.

`sec_chunks` needs no bulk load — filings are fetched, chunked and embedded the
first time a question asks for one.

---

## The daily pipeline

Eight scripts, one Kestra flow, 04:00 UK time. Kestra does not run the scripts
itself: it calls this repository's `/internal/run-pipeline/` endpoints and
polls each one to completion before triggering the next.

**`docs/data-pipeline-reference.md` is the authoritative document for all of
this** — the dependency graph, which orderings are load-bearing and which are
arbitrary, Kestra timeouts and their rationale, the yfinance and FMP
rate-limit findings, the log lines each script emits, and the current known
issues. Read it before changing anything about the pipeline.

The three dependencies that actually constrain the order:

1. `build_stock_universe.py` first — it decides who is in the universe.
2. `update_stock_overviews.py` before `update_industry_tags.py` — tags are
   extracted from the overview text, not from the filing.
3. `update_momentum_benchmarks.py` and `update_financial_history.py` before
   `update_quant_signals.py` — signals are computed from those two tables.

Everything else can be reordered or run alone.

`docs/kestra/equitymind_daily_pipeline.yml` is a copy of the flow definition,
kept here for version control and review. **Editing it changes nothing** — the
flow that runs lives in the Kestra instance and must be edited there.

Registering a new pipeline script takes two edits that must both happen: add it
to `ALLOWED_SCRIPTS` in `api/routes/pipeline.py`, and add the trigger/wait task
pair to the Kestra flow. Push the code first, wait for the redeploy, then add
the Kestra tasks — the other order gives a 404 on the first run.

---

## Running locally

```bash
uvicorn api.main:app --reload
```

- WebSocket: `ws://localhost:8000/api/v1/query/stream`
- Sync query: `POST http://localhost:8000/api/v1/query/sync`

`tests/websocket_debug_tool.py` connects to the WebSocket endpoint without a
browser, which is the quickest way to exercise the full streaming path. Batch
experiments can skip the API entirely and invoke the graph directly, or use
`/query/sync`.

Several Discovery modules have a `__main__` block for exercising one node in
isolation — `discovery_preparation.py` takes an `--isolate` flag that clears
conversation state between questions, which is what a repeated single-turn test
needs and exactly wrong for a multi-turn one.

---

## Tests

```bash
pytest
```

113 tests, all passing as of 2026-08-28. The suite takes about three and a half
minutes. `tests/test_quant.py` covers the signal functions,
`tests/test_nodes.py` the graph nodes, `tests/test_api.py` the HTTP layer.

---

## Conventions

Established over the life of the project, and worth keeping:

- **A failure must never be recorded in a form indistinguishable from a
  success.** This one has been paid for three times: `update_peer_groups.py`
  writing `[]` for both "no peers exist" and "the request failed", so a
  transient error became permanent; `build_stock_universe.py` treating a
  rate-limited fetch as "did not make the top 250", which deleted fourteen
  mega-caps along with their overviews, tags and peers; and the pipeline
  endpoint returning `status: success` for a run whose every request came back
  401. A step that records what its output was derived from can recover on its
  own; a step that just overwrites cannot.
- **Anything expressible as deterministic code is not delegated to a prompt.**
  The LLM's job is to phrase results, not to compute them, and not to normalise
  a fixed shape that code can normalise exactly.
- **Diagnosable failures beat invisible waste.** An empty scope produces a
  visibly incomplete report and a user who complains; nine unnecessary signal
  fetches produce eighteen silent seconds and a larger bill.
- **Bounded failure modes beat unbounded ones.** Exact tag lookup fails as a
  known gap. Similarity ranking fails as "no threshold worked."
- **Claims trace to a primary source** — the filing text, the code, a query
  result. Not to recollection.
- **Imperative prompt headings, not descriptive ones.** Sections phrased as
  descriptions get ignored; sections phrased as instructions
  (`SELECTION SCOPE (say this in your answer):`) are obeyed. And changing a
  rule without changing the worked examples beneath it changes nothing — the
  examples win.
- Code and comments in English throughout.
- Patches are applied as Python here-documents with an `assert old in s` guard
  and an immediate `ast.parse` check.

---

## Not in this repository

`research/` is local-only and not committed. It holds the experiment scripts,
frozen baselines and result files behind the design decisions above —
including the embedding snapshot taken before `overview_embedding` was dropped,
which is the only copy that exists.

<!--
Known documentation drift, found 2026-08-28. None of it affects behaviour;
all of it misleads a reader:

- init_db_stock_universe.py says the table is updated "every few months" —
  daily since 2026-08-14.
- filing_check.py says files_20f_only serves "sector overview embeddings" —
  embeddings were dropped 2026-08-17; the real reason is that a 20-F has no
  Item 1 to summarise at all.
- update_momentum_benchmarks.py suggests running monthly and refers to
  peer_benchmarks.json, which no longer exists.
- update_common_names.py says it does not touch updated_at; that column was
  dropped 2026-08-17.
- init_db_quant_signals.py says quality_period_end lets the updater skip
  recomputing Quality — that skip was removed 2026-08-10 and every ticker is
  recomputed on every run.

Also worth a look: update_common_names.py uses LLM_MODEL for
"Tesla, Inc." -> "Tesla". update_industry_tags.py uses LLM_MODEL_LIGHT for a
comparable restatement task and says why. If the difference is unintentional,
the light model is the right one here.

Delete this comment block once the above are addressed or recorded elsewhere.
-->
