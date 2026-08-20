"""
src/evaluation/metrics.py — Offline evaluation harness (Q4).

Implements:
  - AUC, MRR, nDCG@5, nDCG@10   (ranking metrics)
  - Intra-list diversity, novelty, coverage  (beyond-accuracy)
  - Bootstrap 95% confidence intervals
  - Slicing: cold-start vs warm, head vs tail

Usage
-----
    from src.evaluation.metrics import EvaluationHarness

    harness = EvaluationHarness(dataset="mind")
    report  = harness.run(val_df, retriever_fn, k=10)
    harness.print_report(report)
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Core ranking metrics
# ═══════════════════════════════════════════════════════════════════════════════

def auc(labels: list[int], scores: list[float]) -> float:
    """AUC-ROC. Requires at least one positive and one negative label."""
    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def mrr(labels: list[int]) -> float:
    """Mean Reciprocal Rank — position of the first relevant item."""
    for i, l in enumerate(labels):
        if l == 1:
            return 1.0 / (i + 1)
    return 0.0


def dcg(labels: list[int], k: int) -> float:
    return sum(
        (2 ** l - 1) / np.log2(i + 2)
        for i, l in enumerate(labels[:k])
    )


def ndcg_at_k(labels: list[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank k."""
    ideal = sorted(labels, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(labels, k) / idcg if idcg > 0 else 0.0


def score_impression(labels: list[int]) -> dict:
    """Compute all ranking metrics for a single impression."""
    # Assign descending scores to simulate ranked list
    scores = list(range(len(labels), 0, -1))
    return {
        "auc":     auc(labels, scores),
        "mrr":     mrr(labels),
        "ndcg@5":  ndcg_at_k(labels, 5),
        "ndcg@10": ndcg_at_k(labels, 10),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Beyond-accuracy metrics
# ═══════════════════════════════════════════════════════════════════════════════

def intra_list_diversity(recommended_ids: list[str], embeddings: np.ndarray, id_to_idx: dict) -> float:
    """Average pairwise cosine distance among recommended articles."""
    idxs = [id_to_idx[aid] for aid in recommended_ids if aid in id_to_idx]
    if len(idxs) < 2:
        return 0.0
    vecs = embeddings[idxs]
    # Cosine similarity → distance
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_norm = vecs / np.where(norms == 0, 1, norms)
    sim_matrix = vecs_norm @ vecs_norm.T
    n = len(idxs)
    dist_sum = (n * n - np.trace(sim_matrix) - sim_matrix.sum()) / 2
    pairs = n * (n - 1) / 2
    return float(dist_sum / pairs) if pairs > 0 else 0.0


def novelty(recommended_ids: list[str], article_popularity: dict[str, int], total_users: int) -> float:
    """Self-information novelty: average -log2(P(article))."""
    scores = []
    for aid in recommended_ids:
        pop = article_popularity.get(aid, 1)
        p = pop / total_users
        scores.append(-np.log2(p + 1e-10))
    return float(np.mean(scores)) if scores else 0.0


def coverage(all_recommended: list[str], catalog_size: int) -> float:
    """Fraction of the total catalog ever recommended."""
    return len(set(all_recommended)) / catalog_size if catalog_size > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """
    Returns (mean, lower_bound, upper_bound) for a 95% bootstrap CI.
    """
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(values)
    means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    return float(np.mean(arr)), float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


# ═══════════════════════════════════════════════════════════════════════════════
# Slicing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def slice_cold_warm(df: pd.DataFrame, threshold: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split impressions into cold-start (< threshold clicks) and warm."""
    n_clicks = df["click_history"].apply(len)
    return df[n_clicks < threshold].copy(), df[n_clicks >= threshold].copy()


def slice_head_tail(df: pd.DataFrame, article_popularity: dict[str, int], top_frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split based on whether the clicked article is a head or tail item."""
    sorted_pop = sorted(article_popularity.values(), reverse=True)
    threshold_idx = max(1, int(len(sorted_pop) * top_frac))
    threshold_count = sorted_pop[threshold_idx - 1]

    def is_head(candidates, labels):
        for cid, lbl in zip(candidates, labels):
            if lbl == 1:
                return article_popularity.get(cid, 0) >= threshold_count
        return False

    mask = df.apply(lambda r: is_head(r["candidates"], r["labels"]), axis=1)
    return df[mask].copy(), df[~mask].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation harness
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_df(df: pd.DataFrame, label: str = "all") -> dict:
    """
    Compute ranking metrics + bootstrap CIs over an impressions DataFrame.
    Assumes df has columns: candidates (list), labels (list of 0/1).
    """
    metric_lists: dict[str, list] = {"auc": [], "mrr": [], "ndcg@5": [], "ndcg@10": []}

    for _, row in df.iterrows():
        labels = row.get("labels", [])
        if not labels or all(l is None for l in labels):
            continue
        labels = [int(l) for l in labels if l is not None]
        if not any(labels):
            continue

        m = score_impression(labels)
        for key in metric_lists:
            v = m[key]
            if not np.isnan(v):
                metric_lists[key].append(v)

    report = {"slice": label, "n": len(df)}
    for key, values in metric_lists.items():
        mean, lo, hi = bootstrap_ci(values)
        report[key] = round(mean, 4)
        report[f"{key}_ci"] = f"[{lo:.4f}, {hi:.4f}]"

    return report


def print_report(reports: list[dict]):
    """Pretty-print a list of slice reports."""
    metrics = ["auc", "mrr", "ndcg@5", "ndcg@10"]
    header = f"{'Slice':<20} {'N':>8}  " + "  ".join(f"{m:>10}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in reports:
        row = f"{r['slice']:<20} {r['n']:>8}  "
        row += "  ".join(f"{r.get(m, float('nan')):>10.4f}" for m in metrics)
        print(row)
        # Print CIs below each slice
        ci_row = f"{'':20} {'':8}  "
        ci_row += "  ".join(f"{r.get(m+'_ci', ''):>10}" for m in metrics)
        print(ci_row)
    print("=" * len(header))


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    parser.add_argument("--split",   default="val")
    args = parser.parse_args()

    from src.pipeline.feature_store import load_split

    val_df = load_split(args.dataset, args.split)

    # Overall
    reports = [evaluate_df(val_df, label="all")]

    # Cold vs warm
    cold, warm = slice_cold_warm(val_df)
    reports.append(evaluate_df(cold, label="cold-start"))
    reports.append(evaluate_df(warm, label="warm"))

    print_report(reports)
