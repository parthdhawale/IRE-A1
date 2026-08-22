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
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)

PROC_DIRS = {
    "mind": Path("data/mind/processed"),
    "ebnerd": Path("data/ebnerd/processed"),
}

ENGLISH_STOPWORDS = {
    "the","a","an","is","in","on","at","to","of","and","or","for","with",
    "this","that","it","as","be","was","are","were","has","have","had",
    "i","you","he","she","we","they","its","our","their","by","from","not",
}

# EB-NeRD's articles are Danish — without these, common function words like
# "og" (and) / "er" (is) / "det" (it) survive tokenization untouched, appear
# in 50-97% of the entire corpus (verified: "og" alone is in 96.6% of all
# 125,541 EB-NeRD articles), and turn every query's BM25 score row into a
# near-dense vector — sparse matrix operations then cost nearly as much as
# dense ones despite the corpus/index being represented sparsely.
DANISH_STOPWORDS = {
    "og","i","jeg","det","at","en","den","til","er","som","på","de","med",
    "han","af","for","ikke","der","var","mig","sig","men","et","har","om",
    "vi","min","havde","ham","hun","nu","over","da","fra","du","ud","sin",
    "dem","os","op","man","hans","hvor","eller","hvad","skal","selv","her",
    "alle","vil","blev","kunne","ind","når","være","dog","noget","ville",
    "jo","deres","efter","ned","skulle","denne","end","dette","mit","også",
    "under","have","dig","anden","hende","mine","alt","meget","sit","sine",
    "vor","mod","disse","hvis","din","nogle","hos","blive","mange","ad",
    "bliver","hendes","været","thi","jer","sådan",
    # Beyond the canonical (Snowball) list above: these are near-universal in
    # this specific news corpus (modal verbs / temporal fillers) but not part
    # of the standard stopword list — verified empirically via document
    # frequency (each appears in 35-65% of all EB-NeRD articles).
    "så","kan","ved","år","få",
}

STOPWORDS = ENGLISH_STOPWORDS | DANISH_STOPWORDS


def tokenize(text: str) -> list[str]:
    """Lowercase, split on word boundaries, remove stopwords."""
    if not isinstance(text, str):
        return []
    # Include Danish æ/ø/å — an ASCII-only [a-z] class silently splits Danish
    # words around them into meaningless fragments (e.g. "være" -> "re",
    # "også" -> "ogs"), which both loses real lexical signal and pollutes the
    # vocabulary with garbage high-document-frequency "terms".
    return [t for t in re.findall(r"[a-zæøå]{2,}", text.lower()) if t not in STOPWORDS]


class FastBM25:
    """
    Inverted-index BM25 with vectorized batch scoring via scipy sparse matrices.
    
    Building the index: O(N * avg_doc_len)
    Scoring one query:  O(|query_terms| * avg_posting_list_len)   <- very fast
    Batch scoring M queries: done via sparse matrix multiply, ~instant.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avgdl = self.doc_len.mean() if self.corpus_size > 0 else 1.0

        log.info("    Building vocabulary & inverted index...")
        # Term → integer ID
        self.term2id: dict[str, int] = {}
        doc_tf: list[dict[int, int]] = []  # for each doc: {term_id: count}
        df: dict[int, int] = {}            # document frequency per term

        for doc in corpus:
            counts: dict[int, int] = {}
            for term in doc:
                if term not in self.term2id:
                    self.term2id[term] = len(self.term2id)
                tid = self.term2id[term]
                counts[tid] = counts.get(tid, 0) + 1
            doc_tf.append(counts)
            for tid in counts:
                df[tid] = df.get(tid, 0) + 1

        V = len(self.term2id)
        N = self.corpus_size
        log.info(f"    Vocabulary size: {V:,} terms, {N:,} docs")

        # Pre-compute IDF for each term
        self.idf = np.zeros(V, dtype=np.float32)
        for tid, d in df.items():
            self.idf[tid] = math.log(((N - d + 0.5) / (d + 0.5)) + 1)

        # Pre-compute BM25 TF weights and store as sparse matrix (docs × terms)
        log.info("    Building sparse TF-weight matrix...")
        rows, cols, vals = [], [], []
        # Norm factor per doc
        norm = self.k1 * (1 - b + b * (self.doc_len / self.avgdl))  # shape (N,)

        for doc_idx, counts in enumerate(doc_tf):
            for tid, tf in counts.items():
                # BM25 TF component: tf*(k1+1) / (tf + norm[doc_idx])
                w = tf * (k1 + 1) / (tf + norm[doc_idx])
                rows.append(doc_idx)
                cols.append(tid)
                vals.append(w)

        # Shape: (N_docs, V_vocab)  — multiply by idf vector to get BM25 score
        self.tf_matrix = csr_matrix(
            (vals, (rows, cols)), shape=(N, V), dtype=np.float32
        )
        log.info("    Sparse matrix built.")

    def _query_matrix(self, query_term_ids: list[list[int]]) -> csr_matrix:
        """Build the (Q, V) sparse query matrix (unique terms × IDF weight)."""
        Q = len(query_term_ids)
        V = len(self.term2id)
        q_rows, q_cols = [], []
        for qi, tids in enumerate(query_term_ids):
            for tid in set(tids):  # unique terms per query
                q_rows.append(qi)
                q_cols.append(tid)
        if not q_rows:
            return csr_matrix((Q, V), dtype=np.float32)
        q_idf_vals = [self.idf[tid] for tid in q_cols]
        return csr_matrix((q_idf_vals, (q_rows, q_cols)), shape=(Q, V), dtype=np.float32)

    def get_scores_batch(self, query_term_ids: list[list[int]]) -> np.ndarray:
        """
        Score ALL queries at once via sparse matrix multiply.
        Returns (n_queries, corpus_size) DENSE score matrix.

        Only use this for small batches — it densifies, which costs
        O(n_queries * corpus_size). For scoring many queries against a large
        corpus (e.g. recall@K evaluation), use get_scores_batch_sparse()
        instead and read off top-k directly from the sparse result.
        """
        Q = len(query_term_ids)
        q_matrix = self._query_matrix(query_term_ids)
        if q_matrix.nnz == 0:
            return np.zeros((Q, self.corpus_size), dtype=np.float32)
        # Score: q_matrix @ tf_matrix.T  → (Q, N_docs)
        scores = q_matrix.dot(self.tf_matrix.T)
        return scores.toarray()

    def get_scores_batch_sparse(self, query_term_ids: list[list[int]]) -> csr_matrix:
        """Same scoring as get_scores_batch, but returns the (Q, N_docs) sparse
        result directly instead of densifying. A query only scores nonzero
        against documents sharing at least one term — for a real corpus that's
        a tiny fraction of N_docs, so this avoids the O(Q * N_docs) blowup a
        `.toarray()` forces on every chunk when scoring many queries."""
        q_matrix = self._query_matrix(query_term_ids)
        return q_matrix.dot(self.tf_matrix.T).tocsr()

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Score a single query. Returns (corpus_size,) array."""
        tids = [self.term2id[t] for t in query_tokens if t in self.term2id]
        batch = self.get_scores_batch([tids])
        return batch[0]


class BM25Retriever:
    def __init__(self, dataset: str):
        assert dataset in ("mind", "ebnerd"), f"Unknown dataset: {dataset}"
        self.dataset = dataset
        self.proc_dir = PROC_DIRS[dataset]
        self.bm25: FastBM25 | None = None
        self.corpus_ids: list[str] = []
        self._corpus_id_to_idx: dict[str, int] = {}
        self._articles_lookup: dict[str, dict] = {}
        self._index_path = self.proc_dir / "bm25_index.pkl"

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, force: bool = False):
        """Build (or load cached) BM25 index from article title + abstract."""
        articles = pd.read_parquet(self.proc_dir / "articles.parquet")
        self._articles_lookup = {
            row["article_id"]: {"title": row["title"], "abstract": row["abstract"]}
            for _, row in articles.iterrows()
        }
        current_ids = frozenset(articles["article_id"])

        if not force and self._index_path.exists():
            log.info(f"  Loading cached BM25 index from {self._index_path}")
            with open(self._index_path, "rb") as f:
                state = pickle.load(f)
            # articles.parquet can change (e.g. re-running preprocessing after
            # a fix) without the cached index knowing — reusing it silently
            # would score against a stale, mismatched corpus. Only trust the
            # cache if its article set still matches exactly.
            if state.get("article_ids") == current_ids:
                self.bm25 = state["bm25"]
                self.corpus_ids = state["corpus_ids"]
                self._corpus_id_to_idx = {aid: i for i, aid in enumerate(self.corpus_ids)}
                log.info(f"  Loaded index: {len(self.corpus_ids):,} articles")
                return
            log.warning(
                f"  Cached index ({len(state.get('corpus_ids', [])):,} articles) doesn't match "
                f"the current article catalog ({len(current_ids):,} articles) — rebuilding."
            )

        self.corpus_ids = articles["article_id"].tolist()
        self._corpus_id_to_idx = {aid: i for i, aid in enumerate(self.corpus_ids)}

        log.info(f"  Building BM25 index over {len(self.corpus_ids):,} articles...")
        corpus_tokens = [
            tokenize(f"{row['title']} {row['abstract']} {row['body']}")
            for _, row in articles.iterrows()
        ]

        self.bm25 = FastBM25(corpus_tokens)

        with open(self._index_path, "wb") as f:
            pickle.dump(
                {"bm25": self.bm25, "corpus_ids": self.corpus_ids, "article_ids": current_ids}, f
            )
        log.info(f"  BM25 index saved to {self._index_path}")

    # ── Query & Retrieve ───────────────────────────────────────────────────────

    def _build_query_tokens(
        self,
        click_history: list[str],
        articles_lookup: dict,
        max_recent: int = 5,
    ) -> list[str]:
        """Concatenate titles of the N most recently clicked articles (as the
        assignment spec suggests) into the query.

        Titles only, and only the last 5 articles — not title+abstract of the
        last 10 (the original design): that produced queries averaging ~120
        unique terms, whose posting-list union covers ~95% of the whole
        125K-article corpus regardless of stopword filtering, making every
        query effectively dense and evaluation intractable at 2M queries.
        A short, title-only query is both the assignment's literal suggestion
        and keeps queries genuinely selective."""
        recent = click_history[-max_recent:]
        titles = [articles_lookup.get(aid, {}).get("title", "") for aid in recent]
        return tokenize(" ".join(titles))

    def retrieve(self, click_history: list[str], k: int = 100) -> list[str]:
        """Return top-k article IDs (from the full corpus) for a single user."""
        if self.bm25 is None:
            raise RuntimeError("Call .build() before .retrieve()")
        query = self._build_query_tokens(click_history, self._articles_lookup)
        if not query:
            return []
        scores = self.bm25.get_scores(query)
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        return [self.corpus_ids[i] for i in top_k_idx]

    def score_candidates(self, click_history: list[str], candidate_ids: list[str]) -> list[float]:
        """Score only a given impression's candidate list (for the eval harness).

        Deliberately does NOT go through get_scores()/get_scores_batch_sparse,
        which score the query against the *entire* corpus (needed for
        recall@K's full-corpus search, but here candidate_ids is already a
        tiny curated set — typically ~9-12 articles). Scoring the whole 125K+
        corpus per impression just to keep a handful of values, done in a
        Python loop over ~2M EB-NeRD impressions, is precisely the "score
        everything, discard 99.99%" pattern already fixed in evaluate() —
        it just moved into this call path instead. Slicing the candidates'
        rows directly out of tf_matrix and dotting against the (tiny) query
        IDF vector computes only what's actually needed.
        """
        if self.bm25 is None:
            raise RuntimeError("Call .build() before .score_candidates()")
        query = self._build_query_tokens(click_history, self._articles_lookup)
        if not query:
            return [0.0] * len(candidate_ids)
        tids = [self.bm25.term2id[t] for t in query if t in self.bm25.term2id]
        if not tids:
            return [0.0] * len(candidate_ids)

        cand_idxs = [self._corpus_id_to_idx.get(cid) for cid in candidate_ids]
        valid_positions = [i for i, idx in enumerate(cand_idxs) if idx is not None]
        if not valid_positions:
            return [0.0] * len(candidate_ids)

        q_idf = np.zeros(len(self.bm25.term2id), dtype=np.float32)
        for tid in set(tids):
            q_idf[tid] = self.bm25.idf[tid]

        valid_idxs = [cand_idxs[i] for i in valid_positions]
        sub_tf = self.bm25.tf_matrix[valid_idxs]  # (n_valid, V) sparse — just these rows
        scores_valid = sub_tf.dot(q_idf)  # small dense (n_valid,) result

        scores = [0.0] * len(candidate_ids)
        for pos, val in zip(valid_positions, scores_valid):
            scores[pos] = float(val)
        return scores

    # ── Evaluate (fully vectorized — handles 30k queries in seconds) ───────────

    def evaluate(
        self,
        val_df: pd.DataFrame,
        k_values: list[int] = (50, 100, 200),
    ) -> dict:
        """
        Compute recall@K for each K value on a validation split.
        Uses fully vectorized sparse matrix batch scoring — no Python loops over queries.
        """
        from tqdm import tqdm

        if self.bm25 is None:
            raise RuntimeError("Call .build() before .evaluate()")

        articles_lookup = self._articles_lookup
        max_k = max(k_values)

        # ── Build query token-ID lists and collect ground truth ─────────────
        log.info(f"  Preparing {len(val_df):,} queries...")
        query_term_ids: list[list[int]] = []
        ground_truths: list[set[str]] = []

        for _, row in tqdm(val_df.iterrows(), total=len(val_df),
                           desc="Preparing queries", ncols=80):
            relevant = {
                cid for cid, lbl in zip(row["candidates"], row["labels"])
                if lbl == 1
            }
            if not relevant:
                continue
            tokens = self._build_query_tokens(row["click_history"], articles_lookup)
            tids = [self.bm25.term2id[t] for t in tokens if t in self.bm25.term2id]
            query_term_ids.append(tids)
            ground_truths.append(relevant)

        if not query_term_ids:
            log.warning("  No valid queries found!")
            return {f"recall@{k}": 0.0 for k in k_values}

        # ── Batch score queries in chunks, staying sparse throughout ──────────
        # A query only scores nonzero against documents sharing at least one
        # term — for a real corpus that's a tiny fraction of N_docs. Densifying
        # each chunk to (chunk, N_docs) before taking top-k (the previous
        # approach) costs O(n_queries * N_docs) regardless of chunk size —
        # ~1.99M queries * 125K docs = ~250 billion cells here, which is why
        # this was projected to take ~2 hours. Reading top-k directly off each
        # sparse row avoids ever materializing that dense matrix.
        CHUNK = 2000  # queries per chunk
        n_queries = len(query_term_ids)
        all_top_ids: list[list[str]] = [None] * n_queries

        log.info(f"  Scoring {n_queries:,} queries in chunks of {CHUNK} (sparse top-k, no densification)...")
        from tqdm import tqdm as _tqdm
        for start in _tqdm(range(0, n_queries, CHUNK), desc="Scoring chunks", ncols=80):
            chunk_qids = query_term_ids[start:start + CHUNK]
            scores = self.bm25.get_scores_batch_sparse(chunk_qids)  # sparse CSR (chunk, N_docs)
            for row_i in range(scores.shape[0]):
                row_start, row_end = scores.indptr[row_i], scores.indptr[row_i + 1]
                idxs = scores.indices[row_start:row_end]
                vals = scores.data[row_start:row_end]
                if len(idxs) == 0:
                    top_idx = idxs
                elif len(idxs) <= max_k:
                    top_idx = idxs[np.argsort(-vals)]
                else:
                    part = np.argpartition(-vals, max_k)[:max_k]
                    top_idx = idxs[part][np.argsort(-vals[part])]
                all_top_ids[start + row_i] = [self.corpus_ids[i] for i in top_idx]

        # ── Compute recall@K ─────────────────────────────────────────────────
        log.info("  Computing recall@K...")
        results = {k: [] for k in k_values}
        for qi, relevant in enumerate(ground_truths):
            retrieved_ids = all_top_ids[qi]
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

    retriever = BM25Retriever(args.dataset)
    retriever.build(force=args.force_rebuild)

    val_df = load_split(args.dataset, "val")
    results = retriever.evaluate(val_df, k_values=args.k)
    print("\nBM25 Recall@K Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")

