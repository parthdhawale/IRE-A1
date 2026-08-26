"""
src/evaluation/benchmark_index.py — ANN index choice benchmark (Q6 engineering analysis).

Q3 permits any ANN structure ("FAISS, ScaNN, or brute-force for small scale").
We ship IndexFlatIP, which is exact brute force. That is a *choice*, and this
script is the evidence for it: it measures what the realistic alternatives
actually buy and cost on our own corpus, rather than asserting that flat is
fine.

Compared, all on the same normalized embedding matrix:
  - IndexFlatIP    exact, no training, no parameters
  - IndexIVFFlat   inverted-file: k-means the vectors into `nlist` cells and
                   probe only the nearest `nprobe` at query time. This is the
                   inverted-index idea from BM25 applied to geometry.
  - IndexHNSWFlat  navigable small-world graph: greedily walk toward closer
                   neighbours, with long-range links for fast traversal.

Reported per index: build time, serialized size, mean/p95 query latency,
throughput, and recall@k measured AGAINST THE EXACT FLAT RESULT (so an
approximate index is scored on whether it returns the same neighbours the
exact search would have, which is the property that actually matters here).

Usage:
    python -m src.evaluation.benchmark_index --dataset mind --n-queries 2000
"""

import argparse
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _percentile_latency(times_ms: list[float]) -> tuple[float, float]:
    arr = np.asarray(times_ms)
    return float(arr.mean()), float(np.percentile(arr, 95))


def _index_size_mb(index) -> float:
    """Serialized on-disk size — the number that matters for deployment."""
    import faiss

    with tempfile.NamedTemporaryFile(suffix=".index", delete=True) as fh:
        faiss.write_index(index, fh.name)
        return Path(fh.name).stat().st_size / 1e6


def _time_search(index, queries: np.ndarray, k: int, batch: int = 1) -> tuple[np.ndarray, list[float]]:
    """Search in fixed-size batches, timing each call.

    batch=1 measures single-query serving latency (what an online recommender
    actually pays); larger batches measure offline throughput.
    """
    all_idx, times_ms = [], []
    for start in range(0, len(queries), batch):
        chunk = queries[start : start + batch]
        t0 = time.perf_counter()
        _, idx = index.search(chunk, k)
        times_ms.append((time.perf_counter() - t0) * 1000 / len(chunk))
        all_idx.append(idx)
    return np.vstack(all_idx), times_ms


def benchmark(dataset: str, n_queries: int = 2000, k: int = 100, seed: int = 42) -> list[dict]:
    import faiss

    from src.retrieval.semantic import SemanticRetriever

    retriever = SemanticRetriever(dataset)
    retriever.build()
    vectors = np.ascontiguousarray(retriever.embeddings.astype(np.float32))
    n, dim = vectors.shape
    log.info(f"  Corpus: {n:,} vectors x {dim}d")

    rng = np.random.default_rng(seed)
    queries = vectors[rng.choice(n, size=min(n_queries, n), replace=False)]

    results: list[dict] = []

    def record(name, index, build_s, exact_idx=None, **extra):
        idx, lat = _time_search(index, queries, k, batch=1)
        mean_ms, p95_ms = _percentile_latency(lat)
        _, batch_lat = _time_search(index, queries, k, batch=len(queries))
        recall = None
        if exact_idx is not None:
            hits = sum(len(set(a) & set(b)) for a, b in zip(idx, exact_idx))
            recall = hits / (len(idx) * k)
        row = {
            "index": name,
            "build_s": round(build_s, 2),
            "size_mb": round(_index_size_mb(index), 1),
            "latency_ms": round(mean_ms, 3),
            "p95_ms": round(p95_ms, 3),
            "batch_qps": round(len(queries) / (sum(batch_lat) * len(queries) / 1000), 0),
            f"recall@{k}_vs_exact": round(recall, 4) if recall is not None else 1.0,
            **extra,
        }
        results.append(row)
        log.info(f"  {row}")
        return idx

    # ── Exact baseline (what we ship) ─────────────────────────────────────────
    t0 = time.perf_counter()
    flat = faiss.IndexFlatIP(dim)
    flat.add(vectors)
    exact_idx = record("IndexFlatIP (exact)", flat, time.perf_counter() - t0)

    # ── IVF: partition the space, probe a few cells ───────────────────────────
    nlist = 1024
    for nprobe in (1, 8, 32):
        t0 = time.perf_counter()
        quantizer = faiss.IndexFlatIP(dim)
        ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(vectors)
        ivf.add(vectors)
        build_s = time.perf_counter() - t0
        ivf.nprobe = nprobe
        record(f"IndexIVFFlat nlist={nlist} nprobe={nprobe}", ivf, build_s, exact_idx)

    # ── HNSW: graph traversal ─────────────────────────────────────────────────
    for ef in (32, 128):
        t0 = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 80
        hnsw.add(vectors)
        build_s = time.perf_counter() - t0
        hnsw.hnsw.efSearch = ef
        record(f"IndexHNSWFlat M=32 efSearch={ef}", hnsw, build_s, exact_idx)

    return results


def print_table(rows: list[dict]):
    cols = ["index", "build_s", "size_mb", "latency_ms", "p95_ms", "batch_qps"]
    recall_col = [c for c in rows[0] if c.startswith("recall@")][0]
    cols.append(recall_col)
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
    parser.add_argument("--n-queries", type=int, default=2000)
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    rows = benchmark(args.dataset, args.n_queries, args.k)
    print_table(rows)
