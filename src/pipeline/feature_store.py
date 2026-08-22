"""
src/pipeline/feature_store.py — Build and load the feature store.

Saves reusable article and user features:
  Article features : article_id → {title, abstract, body, category, published_time}
  Embeddings       : article_id → np.array  (if available)
  User features    : user_id    → {click_history, recency_weights}

Output layout
-------------
data/{dataset}/processed/
    articles.parquet          — deduplicated article feature table
    article_embeddings.npy    — numpy array  (N_articles × D)
    article_id_to_idx.json    — article_id → row index in embeddings array
    user_features.parquet     — per-user aggregated click history + recency
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIND_PROC_DIR = Path("data/mind/processed")
EBNERD_PROC_DIR = Path("data/ebnerd/processed")


# ── Article feature store ─────────────────────────────────────────────────────

def build_article_store(proc_dir: Path):
    """Load the combined articles.parquet (already saved by preprocess_all)."""
    combined_path = proc_dir / "articles.parquet"
    if combined_path.exists():
        articles = pd.read_parquet(combined_path)
        log.info(f"  Article store: {len(articles):,} unique articles from {combined_path}")
        return articles

    log.warning(f"  No articles.parquet found in {proc_dir}, skipping.")
    return None


def build_embedding_index(proc_dir: Path, articles: pd.DataFrame):
    """
    Build article_id → index mapping and save embeddings array.

    If embeddings are already in the articles DataFrame (EB-NeRD),
    extract and save them. Otherwise, create a placeholder for later filling
    by src/retrieval/semantic.py.
    """
    id_to_idx = {aid: i for i, aid in enumerate(articles["article_id"])}

    with open(proc_dir / "article_id_to_idx.json", "w") as f:
        json.dump(id_to_idx, f)
    log.info(f"  Saved article_id_to_idx ({len(id_to_idx):,} entries)")

    # EB-NeRD ships embeddings inside the articles parquet
    emb_col = next(
        (c for c in articles.columns if "embed" in c.lower() or "vector" in c.lower()),
        None,
    )
    if emb_col and articles[emb_col].notna().any():
        embeddings = np.stack(articles[emb_col].apply(
            lambda x: np.array(x, dtype=np.float32) if x is not None else None
        ).dropna().values)
        np.save(proc_dir / "article_embeddings.npy", embeddings)
        log.info(f"  Saved pre-computed embeddings: shape={embeddings.shape}")
    else:
        log.info(
            "  No pre-computed embeddings found — embeddings will be computed "
            "by src/retrieval/semantic.py on first run."
        )


# ── User feature store ────────────────────────────────────────────────────────

def _recency_weights(history: list, decay: float = 0.9) -> list:
    """Assign exponentially decaying weights to click history (most recent = 1.0)."""
    n = len(history)
    return [decay ** (n - 1 - i) for i in range(n)]


def build_user_store(proc_dir: Path):
    """Aggregate per-user click history from the training split."""
    train_path = proc_dir / "train.parquet"
    if not train_path.exists():
        log.warning(f"  train.parquet not found in {proc_dir}, skipping user store.")
        return

    # Aggregate click history per user across all training impressions.
    # click_history in each row already contains history before that impression;
    # the most recent (last) impression's history per user is the final history.
    #
    # This scans row-group batches manually via pyarrow instead of a Polars
    # lazy/streaming groupby: on this machine, Polars' streaming engine's
    # internal scratch memory reliably got the process killed (exit 137) even
    # after switching away from a full-table sort, because only ~5-6GB RAM was
    # actually free (other apps were using the rest of the 16GB). A manual
    # reduction into a plain dict has a small, predictable footprint — roughly
    # one (timestamp, click_history) pair per unique user, nothing more.
    import pyarrow.parquet as pq
    from tqdm import tqdm

    log.info(f"  Aggregating per-user click history from {train_path}...")
    pf = pq.ParquetFile(train_path)
    total_rows = pf.metadata.num_rows
    latest: dict[str, tuple] = {}  # user_id -> (timestamp, click_history)

    with tqdm(total=total_rows, desc="  Scanning train for user history", ncols=80) as bar:
        for batch in pf.iter_batches(columns=["user_id", "timestamp", "click_history"], batch_size=200_000):
            for user_id, ts, history in zip(
                batch.column("user_id").to_pylist(),
                batch.column("timestamp").to_pylist(),
                batch.column("click_history").to_pylist(),
            ):
                prev = latest.get(user_id)
                if prev is None or ts > prev[0]:
                    latest[user_id] = (ts, history)
            bar.update(batch.num_rows)

    user_rows = []
    for user_id, (_, history) in latest.items():
        history = history or []
        user_rows.append({
            "user_id": user_id,
            "click_history": history,
            "recency_weights": _recency_weights(history),
            "n_clicks": len(history),
        })

    user_df = pd.DataFrame(user_rows)
    user_df.to_parquet(proc_dir / "user_features.parquet", index=False)
    log.info(f"  User feature store: {len(user_df):,} users → {proc_dir}/user_features.parquet")
    return user_df


# ── Entry point ───────────────────────────────────────────────────────────────

def build_feature_store(dataset: str = "both"):
    datasets_to_build = []
    if dataset in ("both", "mind"):
        datasets_to_build.append(("MIND", MIND_PROC_DIR))
    if dataset in ("both", "ebnerd"):
        datasets_to_build.append(("EB-NeRD", EBNERD_PROC_DIR))

    for name, proc_dir in datasets_to_build:
        log.info(f"Building feature store for {name}...")
        proc_dir.mkdir(parents=True, exist_ok=True)
        articles = build_article_store(proc_dir)
        if articles is not None:
            build_embedding_index(proc_dir, articles)
        build_user_store(proc_dir)


# ── Convenience loaders (used by retrieval modules) ───────────────────────────

def load_articles(dataset: str) -> pd.DataFrame:
    proc_dir = MIND_PROC_DIR if dataset == "mind" else EBNERD_PROC_DIR
    return pd.read_parquet(proc_dir / "articles.parquet")


def load_embeddings(dataset: str) -> tuple[np.ndarray, dict]:
    proc_dir = MIND_PROC_DIR if dataset == "mind" else EBNERD_PROC_DIR
    embeddings = np.load(proc_dir / "article_embeddings.npy")
    with open(proc_dir / "article_id_to_idx.json") as f:
        id_to_idx = json.load(f)
    return embeddings, id_to_idx


def resolve_split_path(dataset: str, split: str) -> Path:
    """Find the parquet file backing a given dataset split, trying the
    split's own name first then falling back to preprocess_all()'s
    impressions_{split} naming (val→validation, dev→val, impressions_X)."""
    proc_dir = MIND_PROC_DIR if dataset == "mind" else EBNERD_PROC_DIR
    path = proc_dir / f"{split}.parquet"
    if not path.exists():
        alternates = [
            proc_dir / f"impressions_{split}.parquet",
            proc_dir / f"impressions_{'dev' if split == 'val' else split}.parquet",
            proc_dir / f"impressions_{'validation' if split == 'val' else split}.parquet",
        ]
        for alt in alternates:
            if alt.exists():
                path = alt
                break
    return path


def load_split(dataset: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(resolve_split_path(dataset, split))
