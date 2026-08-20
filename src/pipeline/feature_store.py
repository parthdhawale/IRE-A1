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
    """Deduplicate articles from all available splits and save."""
    parts = []
    for fname in ["articles_train.parquet", "articles_dev.parquet",
                  "articles_validation.parquet", "articles_test.parquet"]:
        p = proc_dir / fname
        if p.exists():
            parts.append(pd.read_parquet(p))

    if not parts:
        log.warning(f"  No article parquet files found in {proc_dir}, skipping.")
        return

    articles = pd.concat(parts, ignore_index=True).drop_duplicates("article_id")
    articles = articles.reset_index(drop=True)
    articles.to_parquet(proc_dir / "articles.parquet", index=False)
    log.info(f"  Article store: {len(articles):,} unique articles → {proc_dir}/articles.parquet")
    return articles


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

    train = pd.read_parquet(train_path)

    # Aggregate click history per user across all training impressions
    # click_history in each row already contains history before that impression
    # Use the most recent (last) impression's history per user as final history
    user_rows = []
    for user_id, group in train.sort_values("timestamp").groupby("user_id"):
        last_row = group.iloc[-1]
        history = last_row["click_history"] if isinstance(last_row["click_history"], list) else []
        weights = _recency_weights(history)
        user_rows.append({
            "user_id": user_id,
            "click_history": history,
            "recency_weights": weights,
            "n_clicks": len(history),
        })

    user_df = pd.DataFrame(user_rows)
    user_df.to_parquet(proc_dir / "user_features.parquet", index=False)
    log.info(f"  User feature store: {len(user_df):,} users → {proc_dir}/user_features.parquet")
    return user_df


# ── Entry point ───────────────────────────────────────────────────────────────

def build_feature_store(dataset: str = "both"):
    for name, proc_dir in [("MIND", MIND_PROC_DIR), ("EB-NeRD", EBNERD_PROC_DIR)]:
        if dataset not in ("both", name.lower().replace("-", "")):
            # allow 'mind' and 'ebnerd' as shortcuts
            if not (dataset == "mind" and name == "MIND") and \
               not (dataset == "ebnerd" and name == "EB-NeRD"):
                continue

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


def load_split(dataset: str, split: str) -> pd.DataFrame:
    proc_dir = MIND_PROC_DIR if dataset == "mind" else EBNERD_PROC_DIR
    return pd.read_parquet(proc_dir / f"{split}.parquet")
