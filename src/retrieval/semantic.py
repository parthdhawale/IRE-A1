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

EMBEDDING_MODELS = {
    "mind":   "all-MiniLM-L6-v2",                       # English, fast
    "ebnerd": "paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual (Danish)
}

# Bump whenever the embedding model or the query/passage prefixing scheme
# changes. Mirrors TOKENIZER_VERSION in bm25.py: the cache otherwise only
# fingerprints the article-ID set, so swapping models (different
# dimensionality, different prefix convention) would be invisible to the
# staleness check and a stale, incompatible embeddings array would get
# silently reused.
EMBEDDING_MODEL_VERSION = "minilm-v1"


def uses_e5_prefix(model_name: str) -> bool:
    """Whether this model expects E5's "query: "/"passage: " input prefixes.

    E5 models are trained contrastively to expect a role prefix on every
    input, and encoding without it (or with the wrong role) measurably
    degrades retrieval. Models like MiniLM have no such convention, so
    prefixing them would just corrupt the text — hence the per-model switch
    rather than always prefixing.

    Swapping EMBEDDING_MODELS to an E5 checkpoint therefore needs no other
    code change: the passage/query encoding paths below turn on from this,
    and EMBEDDING_MODEL_VERSION invalidates the old cache. (E5-base was
    benchmarked as a candidate but its comparison run could not be completed
    — see report.txt — so the measured MiniLM setup stays the default.)"""
    return "e5" in model_name.lower()

# How many of a user's most recent clicks to mean-pool into their profile
# vector. None = the entire click history.
#
# This was 10, which measurably threw away signal: on a fixed 20K-impression
# MIND val sample, mean-pooling the full history scored AUC 0.6325 vs 0.6194
# for the last 10 (+0.013). Pooling the last 30/50 landed in between, so the
# gain is monotone in history length rather than an artifact of one cutoff.
#
# Exponential recency weighting was also tried and *hurt* (AUC 0.6235 at a
# 5-click half-life, 0.6316 at 20) — counterintuitive for news, but the flat
# mean is the better user representation here, so we keep it. See report.txt.
USER_HISTORY_LIMIT: int | None = None


class SemanticRetriever:
    def __init__(self, dataset: str):
        assert dataset in ("mind", "ebnerd"), f"Unknown dataset: {dataset}"
        self.dataset = dataset
        self.proc_dir = PROC_DIRS[dataset]
        self.embeddings: np.ndarray | None = None        # passage-side (corpus/candidate index)
        self.query_embeddings: np.ndarray | None = None  # query-side (user profiles)
        self.corpus_ids: list[str] = []
        self.id_to_idx: dict[str, int] = {}
        self._faiss_path     = self.proc_dir / "faiss.index"
        self._emb_path       = self.proc_dir / "article_embeddings.npy"
        self._query_emb_path = self.proc_dir / "article_query_embeddings.npy"
        self._ids_path       = self.proc_dir / "corpus_ids.npy"
        self._version_path   = self.proc_dir / "embedding_model_version.txt"

    # ── Build ──────────────────────────────────────────────────────────────────

    def _encode(self, model, texts: list[str], prefix: str | None) -> np.ndarray:
        """Encode texts, optionally under an E5 "query: "/"passage: " prefix."""
        payload = [f"{prefix}: {t}" for t in texts] if prefix else texts
        embeddings = model.encode(
            payload,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via inner product
        )
        return embeddings.astype(np.float32)

    def _compute_embeddings(self, articles: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Compute the (passage, query) embedding pair for title + abstract.

        For an E5-style model the two differ: the candidate/corpus FAISS index
        must be built from passage-prefixed embeddings, while a user's profile
        (mean-pooled from their clicked articles) must come from query-prefixed
        embeddings of that same text — reusing one set for both roles would mix
        vector spaces E5 was trained to keep distinct.

        For a model with no prefix convention the two roles are identical, so
        we encode once and return the same array twice rather than paying to
        compute and store a byte-identical duplicate."""
        from sentence_transformers import SentenceTransformer
        model_name = EMBEDDING_MODELS[self.dataset]
        log.info(f"  Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)

        texts = (articles["title"].fillna("") + " " + articles["abstract"].fillna("")).tolist()
        if not uses_e5_prefix(model_name):
            log.info(f"  Encoding {len(texts):,} articles...")
            shared = self._encode(model, texts, None)
            return shared, shared

        log.info(f"  Encoding {len(texts):,} articles (passage side)...")
        passage_embeddings = self._encode(model, texts, "passage")
        log.info(f"  Encoding {len(texts):,} articles (query side)...")
        query_embeddings = self._encode(model, texts, "query")
        return passage_embeddings, query_embeddings

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
        # saved with are still exactly the current catalog, in the same
        # order, AND it was built with the current embedding model/scheme.
        cached_ids = None
        if self._ids_path.exists():
            cached_ids = np.load(self._ids_path, allow_pickle=True).tolist()
        cached_version = self._version_path.read_text().strip() if self._version_path.exists() else None
        cache_is_fresh = cached_ids == self.corpus_ids and cached_version == EMBEDDING_MODEL_VERSION

        # The query-side file only exists for prefix models; when the two
        # roles are identical there is nothing separate to load or require.
        needs_query_side = uses_e5_prefix(EMBEDDING_MODELS[self.dataset])

        if (
            not force
            and cache_is_fresh
            and self._faiss_path.exists()
            and self._emb_path.exists()
            and (self._query_emb_path.exists() or not needs_query_side)
        ):
            log.info(f"  Loading cached FAISS index from {self._faiss_path}")
            self.embeddings = np.load(self._emb_path)
            self.query_embeddings = (
                np.load(self._query_emb_path) if needs_query_side else self.embeddings
            )
            self.index = faiss.read_index(str(self._faiss_path))
            log.info(f"  Loaded: {self.index.ntotal:,} vectors, dim={self.embeddings.shape[1]}")
            return
        if not force and (cached_ids is not None or cached_version is not None) and not cache_is_fresh:
            log.warning(
                f"  Cached embeddings ({len(cached_ids or []):,} articles, "
                f"model={cached_version}) don't match the current article catalog "
                f"({len(self.corpus_ids):,} articles, model={EMBEDDING_MODEL_VERSION}) "
                f"— recomputing."
            )

        # Load pre-computed or compute fresh embeddings
        if (
            not force
            and cache_is_fresh
            and self._emb_path.exists()
            and (self._query_emb_path.exists() or not needs_query_side)
        ):
            log.info(f"  Loading pre-computed embeddings from {self._emb_path}")
            self.embeddings = np.load(self._emb_path).astype(np.float32)
            self.query_embeddings = (
                np.load(self._query_emb_path).astype(np.float32)
                if needs_query_side
                else self.embeddings
            )
            # Ensure L2-normalised for cosine similarity. When the two roles
            # share one array, normalise it once — iterating over both names
            # would rescale the same buffer twice.
            arrays = [self.embeddings]
            if self.query_embeddings is not self.embeddings:
                arrays.append(self.query_embeddings)
            for arr in arrays:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr /= np.where(norms == 0, 1, norms)
        else:
            self.embeddings, self.query_embeddings = self._compute_embeddings(articles)
            np.save(self._emb_path, self.embeddings)
            if needs_query_side:
                np.save(self._query_emb_path, self.query_embeddings)
            np.save(self._ids_path, np.array(self.corpus_ids, dtype=object))
            self._version_path.write_text(EMBEDDING_MODEL_VERSION)
            log.info(f"  Embeddings saved: shape={self.embeddings.shape}")

        # Build FAISS flat inner-product index (= cosine after normalisation)
        dim = self.embeddings.shape[1]
        log.info(f"  Building FAISS index (dim={dim}, n={len(self.corpus_ids):,})...")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        faiss.write_index(self.index, str(self._faiss_path))
        log.info(f"  FAISS index saved to {self._faiss_path}")

    # ── Query & Retrieve ───────────────────────────────────────────────────────

    def _user_embedding(
        self, click_history: list[str], max_recent: int | None = USER_HISTORY_LIMIT
    ) -> np.ndarray | None:
        """Mean-pool the query-prefixed embeddings of clicked articles.

        Uses query_embeddings, not the passage-prefixed self.embeddings that
        back the FAISS/candidate index — a user profile built from clicked
        articles plays the "query" role in retrieval and must be matched
        against passage-side candidate vectors, per E5's training convention.

        max_recent=None pools the user's whole click history; see
        USER_HISTORY_LIMIT for why."""
        recent = click_history if max_recent is None else click_history[-max_recent:]
        idxs = [self.id_to_idx[aid] for aid in recent if aid in self.id_to_idx]
        if not idxs:
            return None
        user_emb = self.query_embeddings[idxs].mean(axis=0)
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

        # One gather + one mat-vec for the whole candidate list, rather than a
        # per-candidate np.dot in a Python loop.
        #
        # Building the row-index array matters as much as the mat-vec itself.
        # Measured on 1,000 MIND impressions (benchmark_retrieval.py):
        #     per-candidate np.dot in a Python loop      0.062 ms
        #     np.full + per-candidate Python fill loop   0.077 ms   <- WORSE
        #     list-comprehension into np.array (below)   0.034 ms   <- shipped
        # The middle version was the first attempt at this optimization and
        # was slower than the loop it replaced — gathering ~40x384 floats
        # copies real memory, so it only pays off once the index array is
        # built in one shot rather than element-by-element from Python.
        rows = np.array([self.id_to_idx.get(cid, -1) for cid in candidate_ids], dtype=np.int64)
        known = rows >= 0
        scores = np.zeros(len(candidate_ids), dtype=np.float32)
        if known.any():
            # Unknown candidates keep score 0.0, as before.
            scores[known] = self.embeddings[rows[known]] @ user_emb
        return scores.tolist()

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
    parser.add_argument(
        "--sample-n", type=int, default=None,
        help="Measure recall@K on a seeded random sample of this many queries "
             "instead of the whole val split. This is a FULL-CORPUS search, so "
             "its cost scales with queries x catalog; on EB-NeRD (2M queries, "
             "125K articles, ~267-article histories) a full pass is multi-hour.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from src.pipeline.feature_store import load_split

    retriever = SemanticRetriever(args.dataset)
    retriever.build(force=args.force_rebuild)

    val_df = load_split(args.dataset, "val")
    if args.sample_n is not None and args.sample_n < len(val_df):
        val_df = val_df.sample(n=args.sample_n, random_state=args.seed).reset_index(drop=True)
        log.info(f"  Sampled {len(val_df):,} queries (seed={args.seed})")
    results = retriever.evaluate(val_df, k_values=args.k)
    print("\nSemantic Recall@K Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
