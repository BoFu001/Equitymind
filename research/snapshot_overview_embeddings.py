"""Freeze overview_embedding + its source overview_text as a research artifact,
before dropping the column from production. See Research Logs 03/04/05."""

import json
import hashlib
from datetime import date
from pathlib import Path

import sys

import numpy as np
import psycopg2 as psycopg

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATABASE_URL

TABLE = "stock_universe"
EMBEDDING_MODEL = "text-embedding-3-small"

OUT_DIR = Path("research/2026-08-17_embedding_snapshot")
STAMP = date.today().isoformat()


def gprint(msg):
    print(f"\033[92m{msg}\033[0m")


def parse_vector(value):
    if value is None:
        result = None
    elif isinstance(value, str):
        result = [float(x) for x in value.strip("[]").split(",")]
    else:
        result = list(value)
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT ticker, overview_text, overview_embedding "
                f"FROM {TABLE} ORDER BY ticker"
            )
            rows = cur.fetchall()

    gprint(f"fetched {len(rows)} rows from {TABLE}")

    tickers, texts, vectors = [], {}, []
    null_embedding, null_text = [], []

    for ticker, overview_text, embedding in rows:
        vec = parse_vector(embedding)
        if vec is None:
            null_embedding.append(ticker)
        else:
            if overview_text is None:
                null_text.append(ticker)
            tickers.append(ticker)
            texts[ticker] = overview_text
            vectors.append(vec)

    if not vectors:
        raise SystemExit("ABORT: no embeddings found — check TABLE name")

    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise SystemExit(f"ABORT: inconsistent dimensions: {dims}")

    dim = dims.pop()
    matrix = np.asarray(vectors, dtype=np.float32)

    if null_embedding:
        gprint(f"WARNING: {len(null_embedding)} rows NULL embedding, excluded: "
               f"{null_embedding[:10]}")
    if null_text:
        gprint(f"WARNING: {len(null_text)} rows have embedding but NULL text — "
               f"snapshot not self-contained for these: {null_text[:10]}")

    npz_path = OUT_DIR / f"overview_embeddings_{STAMP}.npz"
    txt_path = OUT_DIR / f"overview_texts_{STAMP}.json"
    man_path = OUT_DIR / f"overview_embeddings_{STAMP}.manifest.json"

    np.savez_compressed(npz_path, tickers=np.array(tickers), matrix=matrix)
    txt_path.write_text(json.dumps(texts, ensure_ascii=False, indent=2))

    manifest = {
        "created": STAMP,
        "source_table": TABLE,
        "embedding_model": EMBEDDING_MODEL,
        "n_tickers": len(tickers),
        "dim": dim,
        "excluded_null_embedding": null_embedding,
        "rows_with_null_text": null_text,
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "texts_sha256": hashlib.sha256(txt_path.read_bytes()).hexdigest(),
        "purpose": "Frozen baseline for Research Logs 03/04/05. Production "
                   "dropped this column on the same date; this is the only copy.",
    }
    man_path.write_text(json.dumps(manifest, indent=2))

    gprint(f"wrote {npz_path}  [{len(tickers)} x {dim}]")
    gprint(f"wrote {txt_path}")
    gprint(f"wrote {man_path}")


def verify():
    npz_path = OUT_DIR / f"overview_embeddings_{STAMP}.npz"
    man_path = OUT_DIR / f"overview_embeddings_{STAMP}.manifest.json"

    manifest = json.loads(man_path.read_text())
    data = np.load(npz_path, allow_pickle=False)
    tickers, matrix = data["tickers"], data["matrix"]

    assert len(tickers) == manifest["n_tickers"], "ticker count mismatch"
    assert matrix.shape == (manifest["n_tickers"], manifest["dim"]), "shape mismatch"
    assert not np.isnan(matrix).any(), "NaN present"
    assert (np.linalg.norm(matrix, axis=1) > 0).all(), "zero-norm row"

    gprint(f"VERIFIED  {matrix.shape[0]} x {matrix.shape[1]}  "
           f"model={manifest['embedding_model']}  date={manifest['created']}")


if __name__ == "__main__":
    main()
    verify()
