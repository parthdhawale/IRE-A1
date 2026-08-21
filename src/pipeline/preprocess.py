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
    entities         list[str]  (named-entity labels/groups mentioned in the article)
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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

log = logging.getLogger(__name__)

MIND_RAW_DIR = Path("data/mind/raw")
EBNERD_RAW_DIR = Path("data/ebnerd/raw")
MIND_PROC_DIR = Path("data/mind/processed")
EBNERD_PROC_DIR = Path("data/ebnerd/processed")


def _save_parquet_chunked(df: pd.DataFrame, path: Path, desc: str, chunk_size: int = 500_000):
    """Save massive DataFrame to Parquet in chunks to prevent PyArrow OOM spikes."""
    if len(df) == 0:
        df.to_parquet(path, index=False)
        return

    # FAST SCHEMA INFERENCE: only scan the first 1000 rows.
    # Otherwise PyArrow might silently scan 12 million lists to validate types, freezing the app!
    schema = pa.Schema.from_pandas(df.head(1000))
    
    with pq.ParquetWriter(str(path), schema) as writer:
        for i in tqdm(range(0, len(df), chunk_size), desc=desc, ncols=80):
            chunk = df.iloc[i : i + chunk_size]
            table = pa.Table.from_pandas(chunk, schema=schema)
            writer.write_table(table)


def _to_str_list(x) -> list[str]:
    """Safely convert any array-like (list, np.ndarray, None) to list[str]."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    
    # MEMORY FIX: If it's already a list and the first element is a string, 
    # don't recreate the entire list. Recreating 12M lists causes OOM!
    if type(x) is list and (len(x) == 0 or type(x[0]) is str):
        return x
        
    if isinstance(x, (list, np.ndarray)):
        return [str(a) for a in x]
    try:
        return [str(a) for a in x]  # any other iterable
    except TypeError:
        return []


def _parse_mind_entities(raw: str) -> list[str]:
    """Extract entity Labels from MIND's title_entities/abstract_entities JSON column."""
    import json

    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        return [e["Label"] for e in json.loads(raw) if "Label" in e]
    except (json.JSONDecodeError, TypeError):
        return []


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
    df["entities"] = (
        df["title_entities"].apply(_parse_mind_entities)
        + df["abstract_entities"].apply(_parse_mind_entities)
    ).apply(lambda ents: sorted(set(ents)))
    return df[["article_id", "title", "abstract", "body", "category", "entities", "published_time"]]


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
    # Look for files either in a subfolder or in the raw root
    # Since the zip extraction might dump them flat, we try specific subdirs first.
    # We map 'train' -> 'small_train' or 'large_train' based on what exists,
    # or just use the raw dir if the user unzipped manually.
    
    candidates = [
        MIND_RAW_DIR / f"MINDsmall_{split}",
        MIND_RAW_DIR / f"MINDlarge_{split}",
        MIND_RAW_DIR / split,
        MIND_RAW_DIR / f"small_{split}",
        MIND_RAW_DIR / f"large_{split}",
        MIND_RAW_DIR
    ]
    
    split_dir = None
    for p in candidates:
        if (p / "behaviors.tsv").exists() and (p / "news.tsv").exists():
            split_dir = p
            break
            
    if not split_dir:
        raise FileNotFoundError(
            f"MIND split files (news.tsv/behaviors.tsv) not found for split '{split}'.\n"
            "Run build_pipeline.py without --skip-download first."
        )

    articles = _parse_mind_news(split_dir / "news.tsv")
    impressions = _parse_mind_behaviors(split_dir / "behaviors.tsv")
    # MIND's impression_id is only unique *within* a single behaviors.tsv file —
    # train/dev/test each restart numbering from 1, so combining train+dev (as
    # split_all() now does) would silently collide unrelated impressions under
    # the same ID. Prefix train/dev to keep IDs globally unique once combined.
    # The competition "test" split is exempt: Codabench expects the original
    # bare numeric ImpressionID from MINDlarge_test.zip, unprefixed.
    if split != "test":
        impressions["impression_id"] = f"{split}_" + impressions["impression_id"].astype(str)
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
        EBNERD_RAW_DIR / "ebnerd_testset" / split,
    ]
    for p in candidates:
        if (p / "behaviors.parquet").exists() or (p / "history.parquet").exists():
            return p
    raise FileNotFoundError(
        f"EB-NeRD split dir not found for split '{split}'. Tried: {candidates}\n"
        "Run build_pipeline.py without --skip-download first."
    )


def preprocess_ebnerd(bundle: str = "demo", split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse one EB-NeRD split.
    Returns (articles_df, impressions_df) in unified schema.
    
    Real EB-NeRD schema:
      articles: article_id, title, subtitle (=abstract), body, category_str, published_time, ...
      behaviors: impression_id, user_id, impression_time, article_ids_inview, article_ids_clicked, ...
      history:   user_id, article_id_fixed (list of article IDs read before)
    """
    split_dir = _find_ebnerd_split_dir(bundle, split)

    # ── Articles ──────────────────────────────────────────────────────────────
    # articles.parquet lives at the root level, not inside split dirs
    articles_path = split_dir / "articles.parquet"
    if not articles_path.exists():
        # Prefer the bundle-specific catalog (e.g. ebnerd_testset/articles.parquet) —
        # different bundles/time periods can include articles the others don't.
        articles_path = EBNERD_RAW_DIR / f"ebnerd_{bundle}" / "articles.parquet"
    if not articles_path.exists():
        articles_path = EBNERD_RAW_DIR / "articles.parquet"

    articles = pd.read_parquet(articles_path)
    
    # Map EB-NeRD columns → unified schema
    #   subtitle → abstract  (EB-NeRD has no "abstract" field)
    #   category_str → category  (the human-readable category name)
    articles["abstract"] = articles["subtitle"].fillna("") if "subtitle" in articles.columns else ""
    articles["body"] = articles["body"].fillna("") if "body" in articles.columns else ""
    articles["category"] = articles["category_str"] if "category_str" in articles.columns else ""
    articles["entities"] = (
        articles["entity_groups"].apply(_to_str_list) if "entity_groups" in articles.columns
        else [[] for _ in range(len(articles))]
    )
    if "published_time" not in articles.columns:
        articles["published_time"] = pd.NaT
    articles["article_id"] = articles["article_id"].astype(str)
    articles = articles[["article_id", "title", "abstract", "body", "category", "entities", "published_time"]]

    # ── Behaviors / Impressions ───────────────────────────────────────────────
    behaviors = pd.read_parquet(split_dir / "behaviors.parquet")

    from tqdm import tqdm
    tqdm.pandas(desc=f"  EB-NeRD {split} history")

    # ── History (click history per user) ──────────────────────────────────────
    history_path = split_dir / "history.parquet"
    if history_path.exists():
        history = pd.read_parquet(history_path)
        # history has: user_id, article_id_fixed (list of article IDs)
        hist_col = "article_id_fixed" if "article_id_fixed" in history.columns else "article_ids"
        if hist_col in history.columns:
            history = history.rename(columns={hist_col: "click_history"})
            history["click_history"] = history["click_history"].progress_apply(_to_str_list)
            behaviors = behaviors.merge(
                history[["user_id", "click_history"]].drop_duplicates("user_id"),
                on="user_id",
                how="left",
            )
        else:
            behaviors["click_history"] = [[] for _ in range(len(behaviors))]
    else:
        behaviors["click_history"] = [[] for _ in range(len(behaviors))]

    # Fill NaN click_history with empty lists
    tqdm.pandas(desc=f"  EB-NeRD {split} history join fill")
    behaviors["click_history"] = behaviors["click_history"].progress_apply(_to_str_list)

    # ── Map to unified schema ─────────────────────────────────────────────────
    # article_ids_inview → candidates (numpy.ndarray of int32 in parquet)
    # article_ids_clicked → used to build binary labels
    tqdm.pandas(desc=f"  EB-NeRD {split} candidates")
    behaviors["candidates"] = behaviors["article_ids_inview"].progress_apply(_to_str_list)

    if "article_ids_clicked" in behaviors.columns:
        def build_labels(row):
            clicked = set(_to_str_list(row["article_ids_clicked"]))
            return [1 if c in clicked else 0 for c in row["candidates"]]
        tqdm.pandas(desc=f"  EB-NeRD {split} labels")
        behaviors["labels"] = behaviors.progress_apply(build_labels, axis=1)
    else:
        behaviors["labels"] = behaviors["candidates"].apply(lambda x: [None] * len(x))

    # Timestamp
    if "impression_time" in behaviors.columns:
        behaviors["timestamp"] = pd.to_datetime(behaviors["impression_time"], utc=True)
    else:
        behaviors["timestamp"] = pd.NaT
    
    # Construct impressions DataFrame directly from columns to avoid SettingWithCopy and LossySetitem errors
    impressions = pd.DataFrame({
        "impression_id": behaviors["impression_id"].astype(str),
        "user_id": behaviors["user_id"].astype(str),
        "timestamp": behaviors["timestamp"],
        "click_history": behaviors["click_history"],
        "candidates": behaviors["candidates"],
        "labels": behaviors["labels"]
    }, copy=False)

    log.info(f"  EB-NeRD {bundle}/{split}: {len(articles):,} articles, {len(impressions):,} impressions")
    return articles, impressions


# ── Entry point ───────────────────────────────────────────────────────────────

def preprocess_all(dataset: str = "both"):
    """Parse all splits and save to processed/ as parquet."""
    if dataset in ("both", "mind"):
        log.info("Preprocessing MIND...")
        MIND_PROC_DIR.mkdir(parents=True, exist_ok=True)
        all_articles = []
        for split in ("train", "dev"):
            try:
                articles, impressions = preprocess_mind(split)
                all_articles.append(articles)
                impressions.to_parquet(MIND_PROC_DIR / f"impressions_{split}.parquet", index=False)
            except FileNotFoundError as e:
                log.warning(str(e))

        # Official unlabeled Codabench test set (Q5) — only present if downloaded
        # with --large. Kept separate from train/dev: it must never enter the
        # temporal train/val/test split, only used to generate submission files.
        try:
            test_articles, competition_test = preprocess_mind("test")
            all_articles.append(test_articles)
            competition_test.to_parquet(MIND_PROC_DIR / "impressions_competition_test.parquet", index=False)
            log.info(f"  MIND: saved {len(competition_test):,} unlabeled competition-test impressions")
        except FileNotFoundError as e:
            log.info(f"  MIND competition test set not found (need --large): {e}")

        # Merge and deduplicate articles across splits
        if all_articles:
            combined = pd.concat(all_articles, ignore_index=True).drop_duplicates("article_id")
            combined.to_parquet(MIND_PROC_DIR / "articles.parquet", index=False)
            log.info(f"  MIND: saved {len(combined):,} unique articles to articles.parquet")

    if dataset in ("both", "ebnerd"):
        log.info("Preprocessing EB-NeRD...")
        EBNERD_PROC_DIR.mkdir(parents=True, exist_ok=True)

        # Auto-detect the best available bundle by checking which splits exist.
        # Priority: large > small > demo  (prefer the largest available dataset)
        # The large zip extracts as raw/train/ and raw/validation/ directly.
        # The small/demo zips extract as raw/ebnerd_small/train/ etc.
        def _detect_bundle() -> str:
            for bundle in ("large", "small", "demo"):
                for split in ("train", "validation"):
                    try:
                        _find_ebnerd_split_dir(bundle, split)
                        log.info(f"  EB-NeRD: auto-detected bundle='{bundle}'")
                        return bundle
                    except FileNotFoundError:
                        continue
            return "demo"  # fallback

        bundle = _detect_bundle()

        import gc
        all_articles = []
        for split in ("train", "validation"):
            try:
                articles, impressions = preprocess_ebnerd(bundle=bundle, split=split)
                all_articles.append(articles)
                
                # Save chunked to avoid memory spikes
                out_path = EBNERD_PROC_DIR / f"impressions_{split}.parquet"
                _save_parquet_chunked(impressions, out_path, desc=f"  Saving {split}")
                
                # Free memory immediately - these datasets are huge
                del impressions
                gc.collect()
            except FileNotFoundError as e:
                log.warning(str(e))

        # Official unlabeled Codabench test set (Q5) — only present if downloaded
        # with --large (ebnerd_testset.zip). Kept separate from train/validation:
        # it must never enter the temporal train/val/test split, only used to
        # generate submission files.
        try:
            test_articles, competition_test = preprocess_ebnerd(bundle="testset", split="test")
            all_articles.append(test_articles)
            out_path = EBNERD_PROC_DIR / "impressions_competition_test.parquet"
            _save_parquet_chunked(competition_test, out_path, desc="  Saving competition test")
            log.info(f"  EB-NeRD: saved {len(competition_test):,} unlabeled competition-test impressions")
            del competition_test
            gc.collect()
        except FileNotFoundError as e:
            log.info(f"  EB-NeRD competition test set not found (need --large): {e}")

        # Merge and deduplicate articles across splits
        if all_articles:
            combined = pd.concat(all_articles, ignore_index=True).drop_duplicates("article_id")
            combined.to_parquet(EBNERD_PROC_DIR / "articles.parquet", index=False)
            log.info(f"  EB-NeRD: saved {len(combined):,} unique articles to articles.parquet")



