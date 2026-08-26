"""
src/evaluation/benchmark_retrieval.py — scoring-path engineering metrics (Q6).

Measures the serving-side cost of each retriever and quantifies every
optimization applied to the scoring path, each as a controlled before/after on
identical inputs. Functional quality is reported by the Q4 harness; this file
is about latency, throughput, and the cost of the alternatives we rejected.

Measured here:
  1. Per-impression scoring latency (mean / p50 / p95) and throughput, BM25 vs
     semantic — the number that decides whether a 13.5M-impression submission
     takes minutes or hours.
  2. Targeted candidate scoring vs full-corpus scoring. Both produce the same
     ranking of an impression's candidates, but the naive path scores all
     ~130K articles and discards >99.9% of the result.
  3. BM25 query-token memoization on/off.
  4. BM25 reused IDF buffer vs a fresh vocabulary-width allocation per call.
  5. Semantic vectorized gather+matvec vs a per-candidate Python dot loop.

Usage:
    python -m src.evaluation.benchmark_retrieval --dataset mind --n 1000
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

PROC_DIRS = {"mind": Path("data/mind/processed"), "ebnerd": Path("data/ebnerd/processed")}


def _stats(times_ms: list[float]) -> dict:
    a = np.asarray(times_ms)
    return {
        "mean_ms": round(float(a.mean()), 3),
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "throughput_per_s": round(1000.0 / float(a.mean()), 0),
    }


def _sample(dataset: str, n: int) -> pd.DataFrame:
    """Take impressions from the val split (labels irrelevant here — cost only)."""
    batch = next(
        pq.ParquetFile(PROC_DIRS[dataset] / "val.parquet").iter_batches(
            columns=["click_history", "candidates"], batch_size=max(n, 1000)
        )
    ).to_pandas()
    return batch.head(n).reset_index(drop=True)


def _timed(fn, rows) -> list[float]:
    times = []
    for hist, cands in rows:
        t0 = time.perf_counter()
        fn(hist, cands)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def benchmark(dataset: str, n: int) -> list[dict]:
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.semantic import SemanticRetriever

    df = _sample(dataset, n)
    rows = [(list(r["click_history"]), list(r["candidates"])) for _, r in df.iterrows()]
    log.info(f"  {len(rows):,} impressions | mean history={np.mean([len(h) for h,_ in rows]):.1f} "
             f"| mean candidates={np.mean([len(c) for _,c in rows]):.1f}")

    out: list[dict] = []

    # ── 1. Shipped scoring paths ──────────────────────────────────────────────
    bm = BM25Retriever(dataset); bm.build()
    sem = SemanticRetriever(dataset); sem.build()
    for _ in range(50):  # warm caches so we measure steady state, not first-touch
        bm.score_candidates(*rows[0]); sem.score_candidates(*rows[0])

    out.append({"path": "BM25 score_candidates (shipped)", **_stats(_timed(bm.score_candidates, rows))})
    out.append({"path": "Semantic score_candidates (shipped)", **_stats(_timed(sem.score_candidates, rows))})

    # ── 2. Targeted vs full-corpus scoring ────────────────────────────────────
    def bm25_full_corpus(hist, cands):
        query = bm._build_query_tokens(hist, bm._articles_lookup)
        if not query:
            return [0.0] * len(cands)
        scores = bm.bm25.get_scores(query)           # scores ALL articles
        return [float(scores[bm._corpus_id_to_idx[c]]) for c in cands if c in bm._corpus_id_to_idx]

    subset = rows[: max(1, n // 10)]                  # far slower — smaller sample
    out.append({"path": f"BM25 via full-corpus get_scores (rejected, n={len(subset)})",
                **_stats(_timed(bm25_full_corpus, subset))})

    def sem_full_corpus(hist, cands):
        emb = sem._user_embedding(hist)
        if emb is None:
            return [0.0] * len(cands)
        sims = sem.embeddings @ emb                   # scores ALL articles
        return [float(sims[sem.id_to_idx[c]]) for c in cands if c in sem.id_to_idx]

    out.append({"path": f"Semantic via full-corpus matvec (rejected, n={len(subset)})",
                **_stats(_timed(sem_full_corpus, subset))})

    # ── 3. BM25 token memoization off ─────────────────────────────────────────
    class _NoMemo(dict):
        def get(self, k, d=None): return None
        def __setitem__(self, k, v): pass

    bm_nomemo = BM25Retriever(dataset); bm_nomemo.build()
    bm_nomemo._article_token_cache = _NoMemo()
    out.append({"path": "BM25 without token memoization", **_stats(_timed(bm_nomemo.score_candidates, rows))})

    # ── 4. BM25 fresh IDF allocation per call ─────────────────────────────────
    def bm25_fresh_buffer(hist, cands):
        query = bm._build_query_tokens(hist, bm._articles_lookup)
        if not query:
            return [0.0] * len(cands)
        tids = [bm.bm25.term2id[t] for t in query if t in bm.bm25.term2id]
        if not tids:
            return [0.0] * len(cands)
        idxs = [bm._corpus_id_to_idx.get(c) for c in cands]
        valid = [i for i in idxs if i is not None]
        if not valid:
            return [0.0] * len(cands)
        q_idf = np.zeros(len(bm.bm25.term2id), dtype=np.float32)   # O(V) every call
        for t in set(tids):
            q_idf[t] = bm.bm25.idf[t]
        return bm.bm25.tf_matrix[valid].dot(q_idf).tolist()

    out.append({"path": "BM25 with per-call O(V) IDF allocation", **_stats(_timed(bm25_fresh_buffer, rows))})

    # ── 5. Semantic per-candidate Python dot loop ─────────────────────────────
    def sem_python_loop(hist, cands):
        emb = sem._user_embedding(hist)
        if emb is None:
            return [0.0] * len(cands)
        scores = []
        for cid in cands:
            i = sem.id_to_idx.get(cid)
            scores.append(float(np.dot(emb, sem.embeddings[i])) if i is not None else 0.0)
        return scores

    out.append({"path": "Semantic with per-candidate Python dot loop", **_stats(_timed(sem_python_loop, rows))})

    return out


def artifact_report(dataset: str) -> list[dict]:
    """On-disk cost of each index/artifact — the storage side of the choice."""
    d = PROC_DIRS[dataset]
    names = ["bm25_index.pkl", "faiss.index", "article_embeddings.npy", "articles.parquet"]
    return [
        {"artifact": n, "size_mb": round((d / n).stat().st_size / 1e6, 1)}
        for n in names
        if (d / n).exists()
    ]


def print_table(rows: list[dict], key: str):
    cols = [key] + [c for c in rows[0] if c != key]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    print_table(benchmark(args.dataset, args.n), "path")
    print_table(artifact_report(args.dataset), "artifact")
