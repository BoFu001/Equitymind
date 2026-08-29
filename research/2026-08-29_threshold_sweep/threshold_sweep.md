# Threshold Sweep — whole-overview embedding similarity

Snapshot: `overview_embeddings_2026-08-17.npz` · model: `text-embedding-3-small` · 243 companies · 6 queries

Reconstructs and extends the threshold calibration referred to in
Research Log 03 §3.5 and §4.1 as earlier unlogged work. The six
queries are those of the v3 baseline, chosen in Research Log 03 to
span the tag-frequency range of the corpus.

---

## 1. Does the snapshot reproduce the v3 baseline?

The baseline was computed on 2026-08-11 against the live
`overview_embedding` column; the snapshot was frozen on 2026-08-17,
and its manifest does not record which overview version it embedded.
The two are compared here rather than assumed equivalent.

| Query | Same 20 | Same order | First divergence | Baseline top-1 | Recomputed top-1 | Delta |
|---|---|---|---|---|---|---|
| technology | yes | yes | — | 0.3717 | 0.3717 | +0.0000 |
| financial services | yes | yes | — | 0.4230 | 0.4230 | +0.0000 |
| healthcare | yes | yes | — | 0.4155 | 0.4155 | -0.0000 |
| pharmaceuticals | yes | yes | — | 0.4606 | 0.4606 | +0.0000 |
| semiconductors | yes | no | rank 13 | 0.4963 | 0.4963 | +0.0000 |
| robotics | no | no | rank 18 | 0.3940 | 0.3940 | -0.0000 |

### Where the two differ

- **semiconductors** — the same twenty companies, first reordering at rank 13.
- **robotics** — out: UPS (baseline rank 20). In: TEAM (now rank 18, 0.2320).

Every top-1 score is identical to four decimal places, and the
differences that do exist sit at the bottom of the ranking where
scores are closely packed. The corpus is treated as unchanged for
the sweep below, with these exceptions recorded.

---

## 2. A fixed absolute cutoff

`spread` is the range of companies admitted across the six queries
at that one cutoff; `fold` is the ratio between largest and smallest.

| cutoff | technology | financial services | healthcare | pharmaceuticals | semiconductors | robotics | spread | fold |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 109 | 130 | 47 | 106 | 35 | 48 | 35–130 | 3.7× |
| 0.21 | 96 | 114 | 38 | 87 | 30 | 38 | 30–114 | 3.8× |
| 0.22 | 83 | 101 | 31 | 70 | 25 | 27 | 25–101 | 4.0× |
| 0.23 | 71 | 74 | 21 | 60 | 24 | 19 | 19–74 | 3.9× |
| 0.24 | 62 | 63 | 17 | 48 | 23 | 15 | 15–63 | 4.2× |
| 0.25 | 43 | 47 | 14 | 38 | 21 | 9 | 9–47 | 5.2× |
| 0.26 | 32 | 40 | 13 | 34 | 19 | 9 | 9–40 | 4.4× |
| 0.27 | 25 | 32 | 13 | 30 | 19 | 8 | 8–32 | 4.0× |
| 0.28 | 18 | 27 | 9 | 27 | 18 | 7 | 7–27 | 3.9× |
| 0.29 | 14 | 25 | 7 | 21 | 17 | 7 | 7–25 | 3.6× |
| 0.30 | 9 | 23 | 7 | 20 | 15 | 5 | 5–23 | 4.6× |
| 0.31 | 7 | 20 | 6 | 20 | 15 | 4 | 4–20 | 5.0× |
| 0.32 | 4 | 19 | 5 | 20 | 14 | 4 | 4–20 | 5.0× |
| 0.33 | 2 | 14 | 5 | 17 | 12 | 3 | 2–17 | 8.5× |
| 0.34 | 2 | 13 | 5 | 13 | 9 | 2 | 2–13 | 6.5× |
| 0.35 | 1 | 13 | 3 | 12 | 8 | 1 | 1–13 | 13.0× |
| 0.36 | 1 | 12 | 3 | 12 | 4 | 1 | 1–12 | 12.0× |
| 0.37 | 1 | 11 | 3 | 10 | 4 | 1 | 1–11 | 11.0× |
| 0.38 | 0 | 10 | 2 | 10 | 3 | 1 | 0–10 | — |
| 0.39 | 0 | 6 | 2 | 7 | 1 | 1 | 0–7 | — |
| 0.40 | 0 | 4 | 2 | 6 | 1 | 0 | 0–6 | — |
| 0.41 | 0 | 2 | 2 | 4 | 1 | 0 | 0–4 | — |
| 0.42 | 0 | 1 | 0 | 3 | 1 | 0 | 0–3 | — |
| 0.43 | 0 | 0 | 0 | 2 | 1 | 0 | 0–2 | — |
| 0.44 | 0 | 0 | 0 | 2 | 1 | 0 | 0–2 | — |
| 0.45 | 0 | 0 | 0 | 1 | 1 | 0 | 0–1 | — |
| 0.46 | 0 | 0 | 0 | 1 | 1 | 0 | 0–1 | — |
| 0.47 | 0 | 0 | 0 | 0 | 1 | 0 | 0–1 | — |

At 0.32 — the cutoff whose median across the six
queries is closest to ten companies — the queries return 4–20 companies. `technology` returns 4; `pharmaceuticals` returns 20.

---

## 3. A relative cutoff, normalised to each query's own top score

Cutoff is `top1 * ratio`, the dynamic scheme tested in
`discovery_threshold_calibration.py` at ratios 0.45–0.70.
Normalising to top-1 removes the difference in where each query's
scores sit, but not the difference in how they are distributed.

| ratio | technology | financial services | healthcare | pharmaceuticals | semiconductors | robotics | spread | fold |
|---|---|---|---|---|---|---|---|---|
| 0.45 | 150 | 152 | 59 | 89 | 24 | 69 | 24–152 | 6.3× |
| 0.50 | 132 | 112 | 40 | 59 | 21 | 53 | 21–132 | 6.3× |
| 0.55 | 106 | 73 | 22 | 37 | 18 | 30 | 18–106 | 5.9× |
| 0.60 | 80 | 43 | 14 | 27 | 16 | 16 | 14–80 | 5.7× |
| 0.65 | 60 | 30 | 13 | 20 | 14 | 9 | 9–60 | 6.7× |
| 0.70 | 32 | 24 | 7 | 20 | 8 | 8 | 7–32 | 4.6× |
| 0.75 | 18 | 19 | 5 | 12 | 3 | 6 | 3–19 | 6.3× |
| 0.80 | 11 | 13 | 5 | 10 | 1 | 4 | 1–13 | 13.0× |
| 0.85 | 4 | 12 | 3 | 7 | 1 | 2 | 1–12 | 12.0× |
| 0.90 | 2 | 10 | 3 | 3 | 1 | 1 | 1–10 | 10.0× |
| 0.95 | 1 | 4 | 2 | 2 | 1 | 1 | 1–4 | 4.0× |
| 1.00 | 1 | 1 | 1 | 1 | 1 | 1 | 1–1 | 1.0× |

At the highest ratio the original calibration tested (0.70), the six
queries return 7–32 companies — a spread of 4.6×. Normalisation does not make the setting portable.

---

## 4. Why a score does not transfer between queries

| Query | Top-1 score |
|---|---|
| semiconductors | 0.4963 |
| pharmaceuticals | 0.4606 |
| financial services | 0.4230 |
| healthcare | 0.4155 |
| robotics | 0.3940 |
| technology | 0.3717 |

The best match in the weakest query (`technology`, 0.3717) would place as follows in the others:

| Query | Rank that score would hold |
|---|---|
| semiconductors | 3 |
| pharmaceuticals | 10 |
| financial services | 10 |
| healthcare | 3 |
| robotics | 1 |

One score, six meanings. A cosine value carries no interpretation
that survives a change of query — which is exactly what a fixed
cutoff would have to assume of it.

---

## Limitations

- Six queries, chosen to span tag-frequency density rather than
  sampled from a realistic query distribution. This supports a claim
  about the spread between query types, not about frequency under
  real use.
- Companies admitted is a count, not a precision measure. No
  relevance labelling was applied here; whether the admitted
  companies are the right ones is assessed separately in Research
  Logs 03 and 04 by manual review against source filings.
- One embedding model, one snapshot, one universe composition.
- Query embeddings were computed once each, not repeated to check
  run-to-run variation.

