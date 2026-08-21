"""
src/retrieval/semantic.py — Embedding-based ANN candidate retrieval (Q3).

Usage
-----
    from src.retrieval.semantic import SemanticRetriever

    retriever = SemanticRetriever(dataset="mind")
    retriever.build()                           # compute embeddings + FAISS index
    candidates = retriever.retrieve(click_history, k=100)
    recall = retriever.evaluate(val_df, k_values=[50, 100, 200])
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROC_DIRS = {
    "mind": Path("data/mind/processed"),
    "ebnerd": Path("data/ebnerd/processed"),
}

# Model to use for computing embeddings when not pre-provided
EMBEDDING_MODELS = {
    "mind":   "all-MiniLM-L6-v2",                              # English, fast
    "ebnerd": "paraphrase-multilingual-MiniLM-L12-v2",         # Multilingual (Danish)
}


class SemanticRetriever:
    def __init__(self, dataset: str):
        assert dataset in ("mind", "ebnerd"), f"Unknown dataset: {dataset}"
        self.dataset = dataset
        self.proc_dir = PROC_DIRS[dataset]
        self.embeddings: np.ndarray | None = None
        self.corpus_ids: list[str] = []
        self.id_to_idx: dict[str, int] = {}
        self._faiss_path = self.proc_dir / "faiss.index"
        self._emb_path   = self.proc_dir / "article_embeddings.npy"
        self._ids_path   = self.proc_dir / "corpus_ids.npy"

    # ── Build ──────────────────────────────────────────────────────────────────

    def _compute_embeddings(self, articles: pd.DataFrame) -> np.ndarray:
        """Compute sentence embeddings from title + abstract."""
        from sentence_transformers import SentenceTransformer
        model_name = EMBEDDING_MODELS[self.dataset]
        log.info(f"  Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)

        texts = (articles["title"].fillna("") + " " + articles["abstract"].fillna("")).tolist()
        log.info(f"  Encoding {len(texts):,} articles...")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via inner product
        )
        return embeddings.astype(np.float32)

    def build(self, force: bool = False):
        """Build (or load cached) FAISS index."""
        import faiss

        articles = pd.read_parquet(self.proc_dir / "articles.parquet")
        self.corpus_ids = articles["article_id"].tolist()
        self.id_to_idx  = {aid: i for i, aid in enumerate(self.corpus_ids)}

        # A cached index/embeddings array can silently go stale if
        # articles.parquet changes (e.g. re-running preprocessing after a
        # fix) — the row count and row order it was built from may no longer
        # match self.corpus_ids. Only trust the cache if the ids it was
        # saved with are still exactly the current catalog, in the same order.
        cached_ids = None
        if self._ids_path.exists():
            cached_ids = np.load(self._ids_path, allow_pickle=True).tolist()
        cache_is_fresh = cached_ids == self.corpus_ids

        if not force and cache_is_fresh and self._faiss_path.exists() and self._emb_path.exists():
            log.info(f"  Loading cached FAISS index from {self._faiss_path}")
            self.embeddings = np.load(self._emb_path)
            self.index = faiss.read_index(str(self._faiss_path))
            log.info(f"  Loaded: {self.index.ntotal:,} vectors, dim={self.embeddings.shape[1]}")
            return
        if not force and cached_ids is not None and not cache_is_fresh:
            log.warning(
                f"  Cached embeddings ({len(cached_ids):,} articles) don't match the current "
                f"article catalog ({len(self.corpus_ids):,} articles) — recomputing."
            )

        # Load pre-computed or compute fresh embeddings
        if self._emb_path.exists() and not force and cache_is_fresh:
            log.info(f"  Loading pre-computed embeddings from {self._emb_path}")
            self.embeddings = np.load(self._emb_path).astype(np.float32)
            # Ensure L2-normalised for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings /= np.where(norms == 0, 1, norms)
        else:
            self.embeddings = self._compute_embeddings(articles)
            np.save(self._emb_path, self.embeddings)
            np.save(self._ids_path, np.array(self.corpus_ids, dtype=object))
            log.info(f"  Embeddings saved: shape={self.embeddings.shape}")

        # Build FAISS flat inner-product index (= cosine after normalisation)
        dim = self.embeddings.shape[1]
        log.info(f"  Building FAISS index (dim={dim}, n={len(self.corpus_ids):,})...")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        faiss.write_index(self.index, str(self._faiss_path))
        log.info(f"  FAISS index saved to {self._faiss_path}")

    # ── Query & Retrieve ───────────────────────────────────────────────────────

    def _user_embedding(self, click_history: list[str], max_recent: int = 10) -> np.ndarray | None:
        """Mean-pool the embeddings of recently clicked articles."""
        recent = click_history[-max_recent:]
        idxs = [self.id_to_idx[aid] for aid in recent if aid in self.id_to_idx]
        if not idxs:
            return None
        user_emb = self.embeddings[idxs].mean(axis=0)
        norm = np.linalg.norm(user_emb)
        if norm > 0:
            user_emb /= norm
        return user_emb.astype(np.float32)

    def retrieve(self, click_history: list[str], k: int = 100) -> list[str]:
        """Return top-k article IDs (from the full corpus) for a single user."""
        if self.embeddings is None:
            raise RuntimeError("Call .build() before .retrieve()")

        user_emb = self._user_embedding(click_history)
        if user_emb is None:
            return []

        user_emb = user_emb.reshape(1, -1)
        _, top_k_idx = self.index.search(user_emb, k)
        return [self.corpus_ids[i] for i in top_k_idx[0] if i < len(self.corpus_ids)]

    def score_candidates(self, click_history: list[str], candidate_ids: list[str]) -> list[float]:
        """Score only a given impression's candidate list (for the eval harness),
        instead of searching the whole corpus for a top-k list. Score is the
        cosine similarity (inner product of normalised vectors) between the
        user embedding and each candidate's embedding."""
        if self.embeddings is None:
            raise RuntimeError("Call .build() before .score_candidates()")
        user_emb = self._user_embedding(click_history)
        if user_emb is None:
            return [0.0] * len(candidate_ids)
        scores = []
        for cid in candidate_ids:
            idx = self.id_to_idx.get(cid)
            scores.append(float(np.dot(user_emb, self.embeddings[idx])) if idx is not None else 0.0)
        return scores

    # ── Evaluate ───────────────────────────────────────────────────────────────

    def evaluate(self, val_df: pd.DataFrame, k_values: list[int] = (50, 100, 200)) -> dict:
        """
        Compute recall@K on a validation split.
        Fully vectorized — builds all user embeddings at once, then a single FAISS batch search.
        """
        from tqdm import tqdm

        if self.embeddings is None:
            raise RuntimeError("Call .build() before .evaluate()")

        max_k = max(k_values)

        log.info(f"  Building user embeddings for {len(val_df):,} impressions...")
        user_embs = []
        ground_truths = []

        for _, row in tqdm(val_df.iterrows(), total=len(val_df),
                           desc="Building user embeddings", ncols=80):
            relevant = {cid for cid, lbl in zip(row["candidates"], row["labels"]) if lbl == 1}
            if not relevant:
                continue
            emb = self._user_embedding(row["click_history"])
            if emb is None:
                continue
            user_embs.append(emb)
            ground_truths.append(relevant)

        if not user_embs:
            log.warning("  No valid user embeddings found!")
            return {f"recall@{k}": 0.0 for k in k_values}

        # Single batch FAISS search for all users at once
        query_matrix = np.stack(user_embs, axis=0).astype(np.float32)  # (Q, D)
        log.info(f"  Running FAISS batch search for {len(user_embs):,} users (k={max_k})...")
        _, top_indices = self.index.search(query_matrix, max_k)  # (Q, max_k)

        log.info("  Computing recall@K...")
        results = {k: [] for k in k_values}
        for qi, (row_idx, relevant) in enumerate(zip(top_indices, ground_truths)):
            retrieved_ids = [self.corpus_ids[i] for i in row_idx if i < len(self.corpus_ids)]
            for k in k_values:
                recall = len(set(retrieved_ids[:k]) & relevant) / len(relevant)
                results[k].append(recall)

        summary = {}
        for k in k_values:
            mean_recall = float(np.mean(results[k])) if results[k] else 0.0
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

    retriever = SemanticRetriever(args.dataset)
    retriever.build(force=args.force_rebuild)

    val_df = load_split(args.dataset, "val")
    results = retriever.evaluate(val_df, k_values=args.k)
    print("\nSemantic Recall@K Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
