"""
src/pipeline/split.py — Temporal train/val/test split for interaction data.

Q1 requirement: NEVER use random splits for interaction data.
Always split by time: last N days = test, preceding M days = val, rest = train.

Saves:
    data/{dataset}/processed/train.parquet
    data/{dataset}/processed/val.parquet
    data/{dataset}/processed/test.parquet   (carved from val split by time)
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MIND_PROC_DIR = Path("data/mind/processed")
EBNERD_PROC_DIR = Path("data/ebnerd/processed")


def temporal_split(
    df: pd.DataFrame,
    val_days: float = 1.0,
    test_days: float = 1.0,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame of interactions by time.

    Parameters
    ----------
    df          : must contain a timezone-aware datetime column named `timestamp_col`
    val_days    : number of days before test cutoff to use as validation
    test_days   : number of trailing days to use as test

    Returns
    -------
    train, val, test DataFrames
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)

    # Drop rows with missing timestamps before splitting
    n_missing = df[timestamp_col].isna().sum()
    if n_missing > 0:
        log.warning(f"  Dropping {n_missing:,} rows with missing timestamps before split.")
        df = df.dropna(subset=[timestamp_col])

    max_time = df[timestamp_col].max()
    test_cutoff = max_time - pd.Timedelta(days=test_days)
    val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

    train = df[df[timestamp_col] < val_cutoff].reset_index(drop=True)
    val   = df[(df[timestamp_col] >= val_cutoff) & (df[timestamp_col] < test_cutoff)].reset_index(drop=True)
    test  = df[df[timestamp_col] >= test_cutoff].reset_index(drop=True)

    log.info(
        f"  Split: train={len(train):,}  val={len(val):,}  test={len(test):,}"
        f"  (val_cutoff={val_cutoff.date()}, test_cutoff={test_cutoff.date()})"
    )
    return train, val, test


def split_dataset(proc_dir: Path, impressions_file: str, val_days=1.0, test_days=1.0):
    """Load a processed impressions file, apply temporal split, and save."""
    src = proc_dir / impressions_file
    if not src.exists():
        log.warning(f"  File not found, skipping split: {src}")
        return

    df = pd.read_parquet(src)
    train, val, test = temporal_split(df, val_days=val_days, test_days=test_days)

    train.to_parquet(proc_dir / "train.parquet", index=False)
    val.to_parquet(proc_dir / "val.parquet", index=False)
    test.to_parquet(proc_dir / "test.parquet", index=False)
    log.info(f"  Saved train/val/test splits to {proc_dir}")


def split_all(dataset: str = "both"):
    """Apply temporal splits to all processed impression files."""
    if dataset in ("both", "mind"):
        log.info("Splitting MIND impressions...")
        # MIND train split is large enough to carve val/test from
        split_dataset(MIND_PROC_DIR, "impressions_train.parquet", val_days=1.0, test_days=1.0)

    if dataset in ("both", "ebnerd"):
        log.info("Splitting EB-NeRD impressions...")
        split_dataset(EBNERD_PROC_DIR, "impressions_train.parquet", val_days=1.0, test_days=1.0)
