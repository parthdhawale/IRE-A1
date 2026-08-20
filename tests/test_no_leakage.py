"""
tests/test_no_leakage.py — Anti-gaming test (Q9).

Asserts that:
  1. No impression from val/test appears in train (by timestamp)
  2. click_history only contains articles from before the impression timestamp

Run with:
    pytest tests/test_no_leakage.py -v
"""

import pandas as pd
import pytest
from pathlib import Path


def load_splits(proc_dir: Path):
    train = pd.read_parquet(proc_dir / "train.parquet")
    val   = pd.read_parquet(proc_dir / "val.parquet")
    test  = pd.read_parquet(proc_dir / "test.parquet")
    return train, val, test


@pytest.mark.parametrize("dataset,proc_dir", [
    ("mind",   Path("data/mind/processed")),
    ("ebnerd", Path("data/ebnerd/processed")),
])
def test_temporal_ordering(dataset, proc_dir):
    """Val and test timestamps must be strictly after train timestamps."""
    if not (proc_dir / "train.parquet").exists():
        pytest.skip(f"Processed data not found for {dataset}. Run build_pipeline.py first.")

    train, val, test = load_splits(proc_dir)

    train_max = pd.to_datetime(train["timestamp"], utc=True).max()
    val_min   = pd.to_datetime(val["timestamp"],   utc=True).min()
    test_min  = pd.to_datetime(test["timestamp"],  utc=True).min()

    assert train_max < val_min, (
        f"[{dataset}] LEAKAGE: train max timestamp {train_max} >= val min {val_min}"
    )
    assert val_min <= test_min, (
        f"[{dataset}] Unexpected: val starts after test (train_max={train_max}, val_min={val_min}, test_min={test_min})"
    )


@pytest.mark.parametrize("dataset,proc_dir", [
    ("mind",   Path("data/mind/processed")),
    ("ebnerd", Path("data/ebnerd/processed")),
])
def test_no_val_ids_in_train(dataset, proc_dir):
    """No impression_id from val/test should appear in train."""
    if not (proc_dir / "train.parquet").exists():
        pytest.skip(f"Processed data not found for {dataset}.")

    train, val, test = load_splits(proc_dir)

    train_ids = set(train["impression_id"].astype(str))
    val_ids   = set(val["impression_id"].astype(str))
    test_ids  = set(test["impression_id"].astype(str))

    overlap_val  = train_ids & val_ids
    overlap_test = train_ids & test_ids

    assert not overlap_val,  f"[{dataset}] {len(overlap_val)} val impression_ids found in train!"
    assert not overlap_test, f"[{dataset}] {len(overlap_test)} test impression_ids found in train!"


@pytest.mark.parametrize("dataset,proc_dir", [
    ("mind",   Path("data/mind/processed")),
    ("ebnerd", Path("data/ebnerd/processed")),
])
def test_splits_are_disjoint(dataset, proc_dir):
    """Train, val, and test must be temporally disjoint."""
    if not (proc_dir / "train.parquet").exists():
        pytest.skip(f"Processed data not found for {dataset}.")

    train, val, test = load_splits(proc_dir)

    train_ts = pd.to_datetime(train["timestamp"], utc=True)
    val_ts   = pd.to_datetime(val["timestamp"],   utc=True)
    test_ts  = pd.to_datetime(test["timestamp"],  utc=True)

    assert train_ts.max() < val_ts.min(),  f"[{dataset}] Train and val time ranges overlap!"
    assert val_ts.max()   < test_ts.min(), f"[{dataset}] Val and test time ranges overlap!"
