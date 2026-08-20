"""
src/retrieval/bm25.py — BM25 lexical candidate retrieval (Q2).

Usage
-----
    from src.retrieval.bm25 import BM25Retriever

    retriever = BM25Retriever(dataset="mind")
    retriever.build()                          # build inverted index
    candidates = retriever.retrieve(click_history, k=100)
    recall = retriever.evaluate(val_df, k_values=[50, 100, 200])
"""

import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROC_DIRS = {
    "mind": Path("data/mind/processed"),
    "ebnerd": Path("data/ebnerd/processed"),
}


def tokenize(text: str) -> list[str]:
    """Lowercase and split on word boundaries."""
    return re.findall(r"\w+", text.lower()) if isinstance(text, str) else []


class BM25Retriever:
    def __init__(self, dataset: str):
        assert dataset in ("mind", "ebnerd"), f"Unknown dataset: {dataset}"
        self.dataset = dataset
        self.proc_dir = PROC_DIRS[dataset]
        self.bm25 = None
        self.corpus_ids: list[str] = []
        self._index_path = self.proc_dir / "bm25_index.pkl"

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, force: bool = False):
        """Build (or load cached) BM25 index from article title + abstract."""
        if not force and self._index_path.exists():
            log.info(f"  Loading cached BM25 index from {self._index_path}")
            with open(self._index_path, "rb") as f:
                state = pickle.load(f)
            self.bm25 = state["bm25"]
            self.corpus_ids = state["corpus_ids"]
            log.info(f"  Loaded index: {len(self.corpus_ids):,} articles")
            return

        from rank_bm25 import BM25Okapi

        articles = pd.read_parquet(self.proc_dir / "articles.parquet")
        self.corpus_ids = articles["article_id"].tolist()

        log.info(f"  Building BM25 index over {len(self.corpus_ids):,} articles...")
        corpus_tokens = [
            tokenize(f"{row['title']} {row['abstract']}")
            for _, row in articles.iterrows()
        ]
        self.bm25 = BM25Okapi(corpus_tokens)

        with open(self._index_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "corpus_ids": self.corpus_ids}, f)
        log.info(f"  BM25 index saved to {self._index_path}")

    # ── Query & Retrieve ───────────────────────────────────────────────────────

    def _build_query(self, click_history: list[str], articles_lookup: dict, max_recent: int = 5) -> list[str]:
        """Concatenate titles of the N most recent clicked articles as the query."""
        recent = click_history[-max_recent:]
        titles = [articles_lookup.get(aid, {}).get("title", "") for aid in recent]
        return tokenize(" ".join(titles))

    def retrieve(self, click_history: list[str], articles_lookup: dict, k: int = 100) -> list[str]:
        """Return top-k article IDs for a single user."""
        if self.bm25 is None:
            raise RuntimeError("Call .build() before .retrieve()")

        query = self._build_query(click_history, articles_lookup)
        if not query:
            return []

        scores = self.bm25.get_scores(query)
        top_k_idx = scores.argsort()[::-1][:k]
        return [self.corpus_ids[i] for i in top_k_idx]

    # ── Evaluate ───────────────────────────────────────────────────────────────

    def evaluate(self, val_df: pd.DataFrame, k_values: list[int] = (50, 100, 200)) -> dict:
        """
        Compute recall@K for each K value on a validation split.

        val_df must have columns: click_history, candidates, labels
        """
        from src.pipeline.feature_store import load_articles

        articles = load_articles(self.dataset)
        articles_lookup = {
            row["article_id"]: {"title": row["title"], "abstract": row["abstract"]}
            for _, row in articles.iterrows()
        }

        results = {k: [] for k in k_values}
        max_k = max(k_values)

        log.info(f"  Evaluating BM25 on {len(val_df):,} impressions...")
        for _, row in val_df.iterrows():
            # Ground truth: candidates actually clicked
            relevant = [
                cid for cid, lbl in zip(row["candidates"], row["labels"])
                if lbl == 1
            ]
            if not relevant:
                continue

            retrieved = self.retrieve(row["click_history"], articles_lookup, k=max_k)
            retrieved_set_k = {k: set(retrieved[:k]) for k in k_values}

            for k in k_values:
                recall = len(retrieved_set_k[k] & set(relevant)) / len(relevant)
                results[k].append(recall)

        summary = {}
        for k in k_values:
            mean_recall = np.mean(results[k]) if results[k] else 0.0
            summary[f"recall@{k}"] = round(mean_recall, 4)
            log.info(f"  recall@{k} = {mean_recall:.4f}")

        return summary


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], default="mind")
    parser.add_argument("--k", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    from src.pipeline.feature_store import load_split

    retriever = BM25Retriever(args.dataset)
    retriever.build(force=args.force_rebuild)

    val_df = load_split(args.dataset, "val")
    results = retriever.evaluate(val_df, k_values=args.k)
    print("\nBM25 Recall@K Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
