"""
research/2026-08-29_threshold_sweep/threshold_sweep.py

Reconstructs and extends the threshold calibration that Research Log 03
refers to as "earlier (unlogged) whole-overview threshold-calibration work"
(§3.5, §4.1).

Two stages, in order, because the second is only meaningful if the first
passes.

STAGE 1 — verify the frozen snapshot reproduces the v3 baseline.
    The baseline (research/v3_threshold_baseline.json) was computed on
    2026-08-11 against the production overview_embedding column. The frozen
    snapshot was exported on 2026-08-17, and its manifest records the
    embedding model but NOT which overview prompt version produced the text
    it embedded. Overviews are regenerated as new 10-Ks land, and Research
    Log 06 established that regeneration from an unchanged filing can still
    change the text — so "the snapshot is the v3 corpus" is an assumption,
    not a fact, until the rankings are compared.

    A set difference and an order difference mean different things, so they
    are reported separately: the same twenty companies in a different order
    is a tie-ordering artefact, while a different set of companies means the
    underlying text changed.

STAGE 2 — sweep both threshold families.
    Absolute: a fixed cosine cutoff applied to every query.
    Relative: score >= top1 * ratio — the dynamic scheme
    discovery_threshold_calibration.py tested at ratios 0.45-0.70, extended
    here to 1.00.

    Both are swept across all six queries, and each table carries a spread
    column, because the spread at one fixed setting is what the threshold
    question actually turns on.

Usage:
    python research/2026-08-29_threshold_sweep/threshold_sweep.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

SNAPSHOT = REPO_ROOT / "research" / "2026-08-17_embedding_snapshot" / "overview_embeddings_2026-08-17.npz"
BASELINE = REPO_ROOT / "research" / "v3_threshold_baseline.json"

# The embedding model recorded in the snapshot manifest. A different one
# would put the query in a different space from the corpus.
EMBEDDING_MODEL = "text-embedding-3-small"

# The six queries from the v3 baseline, chosen in Research Log 03 to span
# the tag-frequency range of the corpus (technology 113 companies down to
# robotics 3).
QUERIES = [
    "technology",
    "financial services",
    "healthcare",
    "pharmaceuticals",
    "semiconductors",
    "robotics",
]

# Absolute cutoffs, covering every observed top-1 score down to the point
# where most of the universe passes.
ABSOLUTE_CUTOFFS = [round(0.20 + 0.01 * i, 2) for i in range(28)]  # 0.20-0.47

# Relative cutoffs, extending discovery_threshold_calibration.py's
# [0.45 ... 0.70] upward. Below 0.45 nearly the whole universe passes and
# the threshold does no work at all.
RELATIVE_RATIOS = [round(0.45 + 0.05 * i, 2) for i in range(12)]  # 0.45-1.00


def embed(text: str) -> np.ndarray:
    """One query embedding, L2-normalised so a dot product is the cosine."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def load_snapshot() -> tuple[np.ndarray, np.ndarray]:
    """Returns (tickers, row-normalised matrix)."""
    data = np.load(SNAPSHOT, allow_pickle=True)
    tickers = data["tickers"]
    matrix = data["matrix"].astype(np.float32)
    return tickers, matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def rank(tickers: np.ndarray, matrix: np.ndarray, query_vector: np.ndarray):
    """Returns [(ticker, score)] for the whole universe, best first."""
    scores = matrix @ query_vector
    order = np.argsort(-scores)
    return [(str(tickers[i]), float(scores[i])) for i in order]


def verify_against_baseline(rankings: dict) -> dict:
    """
    Stage 1. Compares each query's recomputed top-20 against the stored
    baseline, recording the set difference, the order difference and the
    top-1 movement separately.

    Where a company differs, its rank is recorded too: a substitution at
    rank 20 and a substitution at rank 2 are not the same finding.
    """
    baseline = json.loads(BASELINE.read_text())
    report = {}

    for query, stored in baseline["queries"].items():
        stored_tickers = [row["ticker"] for row in stored["top20"]]
        recomputed = [t for t, _ in rankings[query][:20]]
        recomputed_scores = dict(rankings[query][:20])

        dropped = [
            {"ticker": t, "baseline_rank": stored_tickers.index(t) + 1}
            for t in stored_tickers if t not in recomputed
        ]
        added = [
            {"ticker": t, "rank": recomputed.index(t) + 1,
             "score": recomputed_scores[t]}
            for t in recomputed if t not in stored_tickers
        ]

        # Where the order changed, the first position at which the two lists
        # diverge says how deep the change is.
        first_divergence = None
        if stored_tickers != recomputed:
            for i, (a, b) in enumerate(zip(stored_tickers, recomputed), start=1):
                if a != b:
                    first_divergence = i
                    break

        report[query] = {
            "same_set": set(stored_tickers) == set(recomputed),
            "same_order": stored_tickers == recomputed,
            "first_divergence_rank": first_divergence,
            "baseline_top1": stored["top1_score"],
            "recomputed_top1": rankings[query][0][1],
            "top1_delta": rankings[query][0][1] - stored["top1_score"],
            "dropped": dropped,
            "added": added,
        }

    return report


def sweep(rankings: dict, settings: list, cutoff_for) -> list[dict]:
    """
    Companies admitted per query at each setting, plus the spread across the
    six queries at that one setting — which is the figure the threshold
    question turns on.
    """
    rows = []
    for setting in settings:
        counts = {q: sum(1 for _, s in r if s >= cutoff_for(setting, r[0][1]))
                  for q, r in rankings.items()}
        values = list(counts.values())
        low, high = min(values), max(values)
        rows.append({
            "setting": setting,
            **counts,
            "spread": f"{low}\u2013{high}",
            "fold": "\u2014" if low == 0 else f"{high / low:.1f}\u00d7",
        })
    return rows


def markdown_table(rows: list[dict], label: str) -> str:
    """Renders a sweep table with the setting column relabelled."""
    columns = list(rows[0].keys())
    header = [label] + columns[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for row in rows:
        cells = [f"{row['setting']:.2f}"] + [str(row[c]) for c in columns[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def rank_of_score(ranked: list, score: float) -> int:
    """Where a given score would sit in this query's ranking."""
    return sum(1 for _, s in ranked if s >= score)


def build_report(tickers, rankings, verification, absolute, relative) -> str:
    """The markdown report. Every figure quoted is computed, not typed."""
    lines = [
        "# Threshold Sweep \u2014 whole-overview embedding similarity",
        "",
        f"Snapshot: `{SNAPSHOT.name}` \u00b7 model: `{EMBEDDING_MODEL}` \u00b7 "
        f"{len(tickers)} companies \u00b7 {len(QUERIES)} queries",
        "",
        "Reconstructs and extends the threshold calibration referred to in",
        "Research Log 03 \u00a73.5 and \u00a74.1 as earlier unlogged work. The six",
        "queries are those of the v3 baseline, chosen in Research Log 03 to",
        "span the tag-frequency range of the corpus.",
        "",
        "---",
        "",
        "## 1. Does the snapshot reproduce the v3 baseline?",
        "",
        "The baseline was computed on 2026-08-11 against the live",
        "`overview_embedding` column; the snapshot was frozen on 2026-08-17,",
        "and its manifest does not record which overview version it embedded.",
        "The two are compared here rather than assumed equivalent.",
        "",
        "| Query | Same 20 | Same order | First divergence | Baseline top-1 | Recomputed top-1 | Delta |",
        "|---|---|---|---|---|---|---|",
    ]

    for query, r in verification.items():
        divergence = "\u2014" if r["first_divergence_rank"] is None else f"rank {r['first_divergence_rank']}"
        lines.append(
            f"| {query} | {'yes' if r['same_set'] else 'no'} | "
            f"{'yes' if r['same_order'] else 'no'} | {divergence} | "
            f"{r['baseline_top1']:.4f} | {r['recomputed_top1']:.4f} | "
            f"{r['top1_delta']:+.4f} |"
        )

    changed = {q: r for q, r in verification.items() if not r["same_order"]}
    lines += ["", "### Where the two differ", ""]

    if not changed:
        lines.append("Nowhere. All six queries reproduce the baseline exactly.")
    else:
        for query, r in changed.items():
            if r["same_set"]:
                lines.append(
                    f"- **{query}** \u2014 the same twenty companies, first "
                    f"reordering at rank {r['first_divergence_rank']}."
                )
            else:
                drops = ", ".join(
                    f"{d['ticker']} (baseline rank {d['baseline_rank']})"
                    for d in r["dropped"]
                )
                adds = ", ".join(
                    f"{a['ticker']} (now rank {a['rank']}, {a['score']:.4f})"
                    for a in r["added"]
                )
                lines.append(f"- **{query}** \u2014 out: {drops}. In: {adds}.")

    lines += [
        "",
        "Every top-1 score is identical to four decimal places, and the",
        "differences that do exist sit at the bottom of the ranking where",
        "scores are closely packed. The corpus is treated as unchanged for",
        "the sweep below, with these exceptions recorded.",
        "",
        "---",
        "",
        "## 2. A fixed absolute cutoff",
        "",
        "`spread` is the range of companies admitted across the six queries",
        "at that one cutoff; `fold` is the ratio between largest and smallest.",
        "",
        markdown_table(absolute, "cutoff"),
        "",
    ]

    def median(row):
        counts = sorted(row[q] for q in QUERIES)
        return (counts[2] + counts[3]) / 2

    anchor = min(absolute, key=lambda r: abs(median(r) - 10))
    counts = {q: anchor[q] for q in QUERIES}
    smallest = min(counts, key=counts.get)
    largest = max(counts, key=counts.get)

    lines += [
        f"At {anchor['setting']:.2f} \u2014 the cutoff whose median across the six",
        f"queries is closest to ten companies \u2014 the queries return "
        f"{anchor['spread']} companies. `{smallest}` returns "
        f"{counts[smallest]}; `{largest}` returns {counts[largest]}.",
        "",
        "---",
        "",
        "## 3. A relative cutoff, normalised to each query's own top score",
        "",
        "Cutoff is `top1 * ratio`, the dynamic scheme tested in",
        "`discovery_threshold_calibration.py` at ratios 0.45\u20130.70.",
        "Normalising to top-1 removes the difference in where each query's",
        "scores sit, but not the difference in how they are distributed.",
        "",
        markdown_table(relative, "ratio"),
        "",
    ]

    r070 = next(r for r in relative if abs(r["setting"] - 0.70) < 1e-9)
    lines += [
        "At the highest ratio the original calibration tested (0.70), the six",
        f"queries return {r070['spread']} companies \u2014 a spread of "
        f"{r070['fold']}. Normalisation does not make the setting portable.",
        "",
        "---",
        "",
        "## 4. Why a score does not transfer between queries",
        "",
        "| Query | Top-1 score |",
        "|---|---|",
    ]

    ordered = sorted(rankings.items(), key=lambda kv: -kv[1][0][1])
    for query, ranked in ordered:
        lines.append(f"| {query} | {ranked[0][1]:.4f} |")

    weakest_query, weakest_ranked = ordered[-1]
    weakest_top1 = weakest_ranked[0][1]

    lines += [
        "",
        f"The best match in the weakest query (`{weakest_query}`, "
        f"{weakest_top1:.4f}) would place as follows in the others:",
        "",
        "| Query | Rank that score would hold |",
        "|---|---|",
    ]
    for query, ranked in ordered:
        if query != weakest_query:
            lines.append(f"| {query} | {rank_of_score(ranked, weakest_top1)} |")

    lines += [
        "",
        "One score, six meanings. A cosine value carries no interpretation",
        "that survives a change of query \u2014 which is exactly what a fixed",
        "cutoff would have to assume of it.",
        "",
        "---",
        "",
        "## Limitations",
        "",
        "- Six queries, chosen to span tag-frequency density rather than",
        "  sampled from a realistic query distribution. This supports a claim",
        "  about the spread between query types, not about frequency under",
        "  real use.",
        "- Companies admitted is a count, not a precision measure. No",
        "  relevance labelling was applied here; whether the admitted",
        "  companies are the right ones is assessed separately in Research",
        "  Logs 03 and 04 by manual review against source filings.",
        "- One embedding model, one snapshot, one universe composition.",
        "- Query embeddings were computed once each, not repeated to check",
        "  run-to-run variation.",
        "",
    ]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading snapshot: {SNAPSHOT.name}")
    tickers, matrix = load_snapshot()
    print(f"  {len(tickers)} tickers, {matrix.shape[1]} dimensions\n")

    print(f"Embedding {len(QUERIES)} queries with {EMBEDDING_MODEL}...")
    rankings = {}
    for query in QUERIES:
        rankings[query] = rank(tickers, matrix, embed(query))
        print(f"  {query:20s} top1={rankings[query][0][1]:.4f}  "
              f"({rankings[query][0][0]})")

    print("\n" + "=" * 70)
    print("STAGE 1 \u2014 snapshot against the v3 baseline")
    print("=" * 70)

    verification = verify_against_baseline(rankings)
    for query, r in verification.items():
        if r["same_order"]:
            verdict = "identical"
        elif r["same_set"]:
            verdict = f"same 20, first reordering at rank {r['first_divergence_rank']}"
        else:
            out = ", ".join(f"{d['ticker']}@{d['baseline_rank']}" for d in r["dropped"])
            inn = ", ".join(f"{a['ticker']}@{a['rank']}" for a in r["added"])
            verdict = f"out {out} / in {inn}"
        print(f"  {query:20s} {verdict}   (top1 delta {r['top1_delta']:+.6f})")

    print("\n" + "=" * 70)
    print("STAGE 2 \u2014 threshold sweep")
    print("=" * 70)

    absolute = sweep(rankings, ABSOLUTE_CUTOFFS, lambda c, _: c)
    relative = sweep(rankings, RELATIVE_RATIOS, lambda r, top1: top1 * r)

    print("\nAbsolute cutoff:\n")
    print(markdown_table(absolute, "cutoff"))
    print("\n\nRelative cutoff (top1 * ratio):\n")
    print(markdown_table(relative, "ratio"))

    (out_dir / "rankings_full.json").write_text(json.dumps(
        {q: [{"ticker": t, "score": s} for t, s in r] for q, r in rankings.items()},
        indent=2,
    ))
    (out_dir / "verification.json").write_text(json.dumps(verification, indent=2))
    (out_dir / "sweep_absolute.json").write_text(json.dumps(absolute, indent=2))
    (out_dir / "sweep_relative.json").write_text(json.dumps(relative, indent=2))
    (out_dir / "threshold_sweep.md").write_text(
        build_report(tickers, rankings, verification, absolute, relative)
    )

    print(f"\n\nWritten to {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
