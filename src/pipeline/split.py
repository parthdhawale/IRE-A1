"""
src/pipeline/split.py — Temporal train/val/test split for interaction data.

Q1 requirement: NEVER use random splits for interaction data.
Always split by time: last N days = test, preceding M days = val, rest = train.

Implemented with Polars lazy scans + streaming sinks so multi-GB impression
files (12M+ rows with list columns) are never fully materialized as a pandas
DataFrame of Python objects — that's what was causing OOM kills on the large
EB-NeRD bundle.

Saves:
    data/{dataset}/processed/train.parquet
    data/{dataset}/processed/val.parquet
    data/{dataset}/processed/test.parquet
"""

import logging
from datetime import timedelta
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

MIND_PROC_DIR = Path("data/mind/processed")
EBNERD_PROC_DIR = Path("data/ebnerd/processed")


def _combined_lazyframe(proc_dir: Path, filenames: list[str], timestamp_col: str) -> pl.LazyFrame | None:
    """Lazily scan and vertically concatenate every impressions file that exists.

    Using every available bundle (e.g. both the "train" and "validation" raw
    splits) instead of just one keeps our own temporal split from silently
    discarding half the labelled interaction data.
    """
    frames = [pl.scan_parquet(proc_dir / f) for f in filenames if (proc_dir / f).exists()]
    if not frames:
        return None
    lf = pl.concat(frames, how="vertical_relaxed")
    return lf.filter(pl.col(timestamp_col).is_not_null())


def split_dataset(
    proc_dir: Path,
    impressions_files: list[str],
    val_days: float = 1.0,
    test_days: float = 1.0,
    timestamp_col: str = "timestamp",
):
    """Temporally split one or more processed impressions files and stream the
    result straight to disk (no full in-memory materialization)."""
    lf = _combined_lazyframe(proc_dir, impressions_files, timestamp_col)
    if lf is None:
        log.warning(f"  None of {impressions_files} found in {proc_dir}, skipping split.")
        return

    log.info(f"  Scanning {', '.join(impressions_files)} for max timestamp...")
    max_time = lf.select(pl.col(timestamp_col).max()).collect(engine="streaming")[0, 0]
    test_cutoff = max_time - timedelta(days=test_days)
    val_cutoff = test_cutoff - timedelta(days=val_days)
    log.info(f"  val_cutoff={val_cutoff}  test_cutoff={test_cutoff}")

    splits = {
        "train": lf.filter(pl.col(timestamp_col) < val_cutoff),
        "val": lf.filter((pl.col(timestamp_col) >= val_cutoff) & (pl.col(timestamp_col) < test_cutoff)),
        "test": lf.filter(pl.col(timestamp_col) >= test_cutoff),
    }
    for name, split_lf in splits.items():
        out_path = proc_dir / f"{name}.parquet"
        log.info(f"  Streaming {name} split to {out_path}...")
        split_lf.sink_parquet(out_path)
        n_rows = pq.ParquetFile(out_path).metadata.num_rows
        log.info(f"    {name}: {n_rows:,} rows")

    log.info(f"  Saved train/val/test splits to {proc_dir}")


def split_all(dataset: str = "both"):
    """Apply temporal splits to all processed impression files."""
    if dataset in ("both", "mind"):
        log.info("Splitting MIND impressions...")
        split_dataset(
            MIND_PROC_DIR,
            ["impressions_train.parquet", "impressions_dev.parquet"],
            val_days=1.0,
            test_days=1.0,
        )

    if dataset in ("both", "ebnerd"):
        log.info("Splitting EB-NeRD impressions...")
        split_dataset(
            EBNERD_PROC_DIR,
            ["impressions_train.parquet", "impressions_validation.parquet"],
            val_days=1.0,
            test_days=1.0,
        )
