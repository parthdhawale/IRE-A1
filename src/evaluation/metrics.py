"""
src/evaluation/metrics.py — Offline evaluation harness (Q4).

Implements:
  - AUC, MRR, nDCG@5, nDCG@10   (ranking metrics)
  - Intra-list diversity, novelty, coverage  (beyond-accuracy)
  - Bootstrap 95% confidence intervals
  - Slicing: cold-start vs warm, head vs tail

EvaluationHarness.run() re-ranks each impression's own candidate list using a
retriever's score_candidates(click_history, candidate_ids) (BM25Retriever /
SemanticRetriever both implement this), then scores that ranking — so the
same harness can be pointed at either retriever (Q4.5).

Usage
-----
    from src.evaluation.metrics import EvaluationHarness

    harness = EvaluationHarness(dataset="mind")
    reports = harness.run(val_df, retriever, k=10)
    harness.print_report(reports)
"""

import logging

import numpy as np
import pandas as pd
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


def score_impression(ranked_labels: list[int]) -> dict:
    """Compute all ranking metrics for a single impression, given its labels
    already ordered by a retriever's ranking (most relevant first)."""
    # Descending synthetic scores that encode rank order, for AUC.
    scores = list(range(len(ranked_labels), 0, -1))
    return {
        "auc":     auc(ranked_labels, scores),
        "mrr":     mrr(ranked_labels),
        "ndcg@5":  ndcg_at_k(ranked_labels, 5),
        "ndcg@10": ndcg_at_k(ranked_labels, 10),
    }


def rank_impression(row: pd.Series, retriever) -> tuple[list[int], list[str]]:
    """Re-rank this impression's own candidate list by the retriever's score
    (not a full-corpus top-k search — just orders the candidates actually
    shown). Returns (ranked_labels, ranked_article_ids), most relevant first."""
    candidates = row["candidates"]
    labels = [int(l) if l is not None else 0 for l in row["labels"]]
    scores = retriever.score_candidates(row["click_history"], candidates)
    order = np.argsort(scores)[::-1]
    ranked_ids = [candidates[i] for i in order]
    ranked_labels = [labels[i] for i in order]
    return ranked_labels, ranked_ids


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


def article_popularity(df: pd.DataFrame) -> dict[str, int]:
    """Count how often each article was clicked across an impressions DataFrame."""
    pop: dict[str, int] = {}
    for candidates, labels in zip(df["candidates"], df["labels"]):
        for cid, lbl in zip(candidates, labels):
            if lbl == 1:
                pop[cid] = pop.get(cid, 0) + 1
    return pop


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation harness
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationHarness:
    """Re-ranks impressions with a retriever and reports ranking + beyond-accuracy
    metrics, with slicing and bootstrap CIs. Works with any retriever exposing
    score_candidates(click_history, candidate_ids) -> list[float]
    (BM25Retriever and SemanticRetriever both do)."""

    def __init__(self, dataset: str):
        self.dataset = dataset

    def _diversity_source(self, retriever):
        """Reuse a SemanticRetriever's own embeddings if we were handed one;
        otherwise fall back to whatever's on disk (e.g. from a prior
        semantic.py run) so diversity can still be computed for BM25."""
        embeddings = getattr(retriever, "embeddings", None)
        id_to_idx = getattr(retriever, "id_to_idx", None)
        if embeddings is not None and id_to_idx is not None:
            return embeddings, id_to_idx
        try:
            from src.pipeline.feature_store import load_embeddings
            return load_embeddings(self.dataset)
        except FileNotFoundError:
            return None, None

    def _evaluate_slice(
        self, df: pd.DataFrame, retriever, k: int,
        pop: dict, embeddings, id_to_idx, catalog_size: int, label: str,
    ) -> dict:
        from tqdm import tqdm

        metric_lists: dict[str, list] = {"auc": [], "mrr": [], "ndcg@5": [], "ndcg@10": []}
        diversity_scores, novelty_scores, all_recommended = [], [], []
        total_users = max(len(df), 1)

        log.info(f"  Evaluating slice '{label}' ({len(df):,} impressions)...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {label}", ncols=80):
            # labels/candidates come back as numpy arrays (not plain lists)
            # from pyarrow-backed parquet list columns via df.iterrows() — a
            # bare `not labels` on a multi-element array raises "truth value
            # of an array... is ambiguous", so check length/content explicitly.
            labels = list(row.get("labels", []))
            if len(labels) == 0 or all(l is None for l in labels) or not any(labels):
                continue

            ranked_labels, ranked_ids = rank_impression(row, retriever)
            m = score_impression(ranked_labels)
            for key in metric_lists:
                v = m[key]
                if not np.isnan(v):
                    metric_lists[key].append(v)

            topk_ids = ranked_ids[:k]
            all_recommended.extend(topk_ids)
            novelty_scores.append(novelty(topk_ids, pop, total_users))
            if embeddings is not None:
                diversity_scores.append(intra_list_diversity(topk_ids, embeddings, id_to_idx))

        report = {"slice": label, "n": len(df)}
        for key, values in metric_lists.items():
            mean, lo, hi = bootstrap_ci(values)
            report[key] = round(mean, 4)
            report[f"{key}_ci"] = f"[{lo:.4f}, {hi:.4f}]"

        report["diversity"] = round(float(np.mean(diversity_scores)), 4) if diversity_scores else float("nan")
        report["novelty"] = round(float(np.mean(novelty_scores)), 4) if novelty_scores else float("nan")
        report["coverage"] = round(coverage(all_recommended, catalog_size), 4) if catalog_size else float("nan")
        return report

    def run(
        self, val_df: pd.DataFrame, retriever, k: int = 10,
        slice_by: str = "cold_warm",
    ) -> list[dict]:
        """Re-rank every impression in val_df with `retriever`, then report
        overall metrics plus one slice (cold-start/warm or head/tail)."""
        from src.pipeline.feature_store import load_articles

        catalog_size = len(load_articles(self.dataset))
        pop = article_popularity(val_df)
        embeddings, id_to_idx = self._diversity_source(retriever)

        reports = [self._evaluate_slice(val_df, retriever, k, pop, embeddings, id_to_idx, catalog_size, "all")]

        if slice_by == "cold_warm":
            cold, warm = slice_cold_warm(val_df)
            reports.append(self._evaluate_slice(cold, retriever, k, pop, embeddings, id_to_idx, catalog_size, "cold-start"))
            reports.append(self._evaluate_slice(warm, retriever, k, pop, embeddings, id_to_idx, catalog_size, "warm"))
        elif slice_by == "head_tail":
            head, tail = slice_head_tail(val_df, pop)
            reports.append(self._evaluate_slice(head, retriever, k, pop, embeddings, id_to_idx, catalog_size, "head"))
            reports.append(self._evaluate_slice(tail, retriever, k, pop, embeddings, id_to_idx, catalog_size, "tail"))
        else:
            raise ValueError(f"Unknown slice_by: {slice_by}")

        return reports

    def print_report(self, reports: list[dict]):
        print_report(reports)


def print_report(reports: list[dict]):
    """Pretty-print a list of slice reports (ranking + beyond-accuracy)."""
    ci_cols = ["auc", "mrr", "ndcg@5", "ndcg@10"]
    cols = ci_cols + ["diversity", "novelty", "coverage"]
    header = f"{'Slice':<20} {'N':>8}  " + "  ".join(f"{m:>10}" for m in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in reports:
        row = f"{r['slice']:<20} {r['n']:>8}  "
        row += "  ".join(f"{r.get(m, float('nan')):>10.4f}" for m in cols)
        print(row)
        # Print CIs (ranking metrics only) below each slice
        ci_row = f"{'':20} {'':8}  "
        ci_row += "  ".join(f"{r.get(m + '_ci', ''):>10}" for m in ci_cols)
        print(ci_row)
    print("=" * len(header))


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    parser.add_argument("--split", default="val")
    parser.add_argument("--retriever", choices=["bm25", "semantic"], default="bm25")
    parser.add_argument("--k", type=int, default=10, help="Cutoff for diversity/novelty/coverage")
    parser.add_argument("--slice", choices=["cold_warm", "head_tail"], default="cold_warm")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    from src.pipeline.feature_store import load_split

    if args.retriever == "bm25":
        from src.retrieval.bm25 import BM25Retriever
        retriever = BM25Retriever(args.dataset)
    else:
        from src.retrieval.semantic import SemanticRetriever
        retriever = SemanticRetriever(args.dataset)
    retriever.build(force=args.force_rebuild)

    val_df = load_split(args.dataset, args.split)

    harness = EvaluationHarness(args.dataset)
    reports = harness.run(val_df, retriever, k=args.k, slice_by=args.slice)
    harness.print_report(reports)
