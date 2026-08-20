"""
src/pipeline/preprocess.py — Parse raw MIND + EB-NeRD into a unified schema.

Unified schemas
---------------
Articles DataFrame columns:
    article_id       str
    title            str
    abstract         str
    body             str  (empty string if not available)
    category         str
    published_time   datetime64[ns, UTC]  (NaT if not available)

Impressions DataFrame columns:
    impression_id    str
    user_id          str
    timestamp        datetime64[ns, UTC]
    click_history    list[str]   article IDs clicked before this impression
    candidates       list[str]   article IDs shown in this impression
    labels           list[int]   1 = clicked, 0 = not clicked  (NaN for test set)
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MIND_RAW_DIR = Path("data/mind/raw")
EBNERD_RAW_DIR = Path("data/ebnerd/raw")
MIND_PROC_DIR = Path("data/mind/processed")
EBNERD_PROC_DIR = Path("data/ebnerd/processed")

# ── MIND ──────────────────────────────────────────────────────────────────────

def _parse_mind_news(news_path: Path) -> pd.DataFrame:
    """Parse MIND news.tsv into articles DataFrame."""
    cols = [
        "article_id", "category", "subcategory",
        "title", "abstract", "url",
        "title_entities", "abstract_entities",
    ]
    df = pd.read_csv(news_path, sep="\t", header=None, names=cols)
    df["body"] = ""
    df["published_time"] = pd.NaT
    df["abstract"] = df["abstract"].fillna("")
    return df[["article_id", "title", "abstract", "body", "category", "published_time"]]


def _parse_mind_behaviors(behaviors_path: Path) -> pd.DataFrame:
    """Parse MIND behaviors.tsv into impressions DataFrame."""
    cols = ["impression_id", "user_id", "timestamp", "click_history", "impressions"]
    df = pd.read_csv(behaviors_path, sep="\t", header=None, names=cols)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["click_history"] = df["click_history"].fillna("").apply(
        lambda x: x.split() if x else []
    )

    def parse_impressions(imp_str):
        candidates, labels = [], []
        if pd.isna(imp_str):
            return [], []
        for item in imp_str.split():
            if "-" in item:
                aid, label = item.rsplit("-", 1)
                candidates.append(aid)
                labels.append(int(label))
            else:
                candidates.append(item)
                labels.append(None)  # test set — no labels
        return candidates, labels

    df[["candidates", "labels"]] = df["impressions"].apply(
        lambda x: pd.Series(parse_impressions(x))
    )
    return df[["impression_id", "user_id", "timestamp", "click_history", "candidates", "labels"]]


def preprocess_mind(split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse one MIND split (train / dev).
    Returns (articles_df, impressions_df).
    """
    split_dir = MIND_RAW_DIR / f"MINDsmall_{split}"
    if not split_dir.exists():
        # Try alternate path layout from HF download
        split_dir = MIND_RAW_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"MIND split directory not found: {split_dir}\n"
            "Run build_pipeline.py without --skip-download first."
        )

    articles = _parse_mind_news(split_dir / "news.tsv")
    impressions = _parse_mind_behaviors(split_dir / "behaviors.tsv")
    log.info(f"  MIND {split}: {len(articles):,} articles, {len(impressions):,} impressions")
    return articles, impressions


# ── EB-NeRD ───────────────────────────────────────────────────────────────────

def _find_ebnerd_split_dir(bundle: str, split: str) -> Path:
    """Locate the split directory inside EB-NeRD raw folder."""
    # Common layout: data/ebnerd/raw/ebnerd_<bundle>/<split>/
    candidates = [
        EBNERD_RAW_DIR / f"ebnerd_{bundle}" / split,
        EBNERD_RAW_DIR / bundle / split,
        EBNERD_RAW_DIR / split,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"EB-NeRD split dir not found. Tried: {candidates}\n"
        "Run build_pipeline.py without --skip-download first."
    )


def preprocess_ebnerd(bundle: str = "demo", split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse one EB-NeRD split.
    Returns (articles_df, impressions_df).
    """
    split_dir = _find_ebnerd_split_dir(bundle, split)

    # Articles
    articles = pd.read_parquet(split_dir / "articles.parquet")
    articles = articles.rename(columns={"article_id": "article_id"})
    for col in ["title", "abstract", "body", "category", "published_time"]:
        if col not in articles.columns:
            articles[col] = "" if col != "published_time" else pd.NaT
    articles["abstract"] = articles["abstract"].fillna("")
    articles["body"] = articles["body"].fillna("")
    articles = articles[["article_id", "title", "abstract", "body", "category", "published_time"]]
    articles["article_id"] = articles["article_id"].astype(str)

    # Behaviors / impressions
    behaviors = pd.read_parquet(split_dir / "behaviors.parquet")

    # History file (click history per user at time of impression)
    history_path = split_dir / "history.parquet"
    if history_path.exists():
        history = pd.read_parquet(history_path)
        # history has: user_id, impression_time, article_id_fixed (list)
        history = history.rename(columns={"article_id_fixed": "click_history"})
        history["click_history"] = history["click_history"].apply(
            lambda x: [str(a) for a in x] if isinstance(x, list) else []
        )
        behaviors = behaviors.merge(
            history[["user_id", "click_history"]].drop_duplicates("user_id"),
            on="user_id",
            how="left",
        )
    else:
        behaviors["click_history"] = [[] for _ in range(len(behaviors))]

    # Normalise column names
    rename_map = {
        "impression_id": "impression_id",
        "user_id": "user_id",
        "impression_time": "timestamp",
        "article_id_inview": "candidates",
        "article_ids_clicked": "labels_raw",
    }
    behaviors = behaviors.rename(columns={k: v for k, v in rename_map.items() if k in behaviors.columns})

    if "timestamp" not in behaviors.columns:
        behaviors["timestamp"] = pd.NaT
    else:
        behaviors["timestamp"] = pd.to_datetime(behaviors["timestamp"], utc=True)

    behaviors["candidates"] = behaviors["candidates"].apply(
        lambda x: [str(a) for a in x] if isinstance(x, list) else []
    )

    if "labels_raw" in behaviors.columns:
        def build_labels(row):
            clicked = set(str(a) for a in (row["labels_raw"] or []))
            return [1 if str(c) in clicked else 0 for c in row["candidates"]]
        behaviors["labels"] = behaviors.apply(build_labels, axis=1)
    else:
        behaviors["labels"] = behaviors["candidates"].apply(lambda x: [None] * len(x))

    impressions = behaviors[["impression_id", "user_id", "timestamp", "click_history", "candidates", "labels"]]
    impressions = impressions.copy()
    impressions["impression_id"] = impressions["impression_id"].astype(str)
    impressions["user_id"] = impressions["user_id"].astype(str)

    log.info(f"  EB-NeRD {bundle}/{split}: {len(articles):,} articles, {len(impressions):,} impressions")
    return articles, impressions


# ── Entry point ───────────────────────────────────────────────────────────────

def preprocess_all(dataset: str = "both"):
    """Parse all splits and save to processed/ as parquet."""
    if dataset in ("both", "mind"):
        log.info("Preprocessing MIND...")
        MIND_PROC_DIR.mkdir(parents=True, exist_ok=True)
        for split in ("train", "dev"):
            try:
                articles, impressions = preprocess_mind(split)
                articles.to_parquet(MIND_PROC_DIR / f"articles_{split}.parquet", index=False)
                impressions.to_parquet(MIND_PROC_DIR / f"impressions_{split}.parquet", index=False)
            except FileNotFoundError as e:
                log.warning(str(e))

    if dataset in ("both", "ebnerd"):
        log.info("Preprocessing EB-NeRD...")
        EBNERD_PROC_DIR.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation"):
            try:
                articles, impressions = preprocess_ebnerd(bundle="demo", split=split)
                articles.to_parquet(EBNERD_PROC_DIR / f"articles_{split}.parquet", index=False)
                impressions.to_parquet(EBNERD_PROC_DIR / f"impressions_{split}.parquet", index=False)
            except FileNotFoundError as e:
                log.warning(str(e))
