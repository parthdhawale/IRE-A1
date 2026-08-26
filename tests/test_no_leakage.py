"""
tests/test_no_leakage.py — Anti-gaming test (Q9).

Asserts that:
  1. Train/val/test are temporally ordered and disjoint (no impression from
     val/test appears in train, by id or by timestamp)
  2. The behaviour-window boundary holds: an impression's click_history never
     contains a click that happened after that impression. This is the
     assignment's explicit "no future-click leakage" requirement, and it is a
     SEPARATE property from (1) — the splits can be perfectly ordered while
     the features still leak, if history were built by aggregating all of a
     user's clicks regardless of when the impression occurred.

     It is checked two ways, because the datasets expose different evidence:
       - EB-NeRD ships published_time for every article, so we assert
         directly that no article in a history was published after the
         impression that carries it.
       - MIND's news.tsv has no publication timestamp at all (published_time
         is null for all 130,379 articles), so instead we assert the
         equivalent structural property: for a single user, history can only
         grow over time. If a future click had leaked backwards into an
         earlier impression, that impression's history would contain an
         article missing from a later one, breaking monotonicity.

Run with:
    pytest tests/test_no_leakage.py -v
"""

import itertools
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest


def _as_naive(value) -> pd.Timestamp:
    """Drop tz info so parquet-read timestamps compare consistently.

    articles.parquet and the impression splits do not agree on tz-awareness,
    and comparing a tz-aware to a tz-naive Timestamp raises rather than
    returning False — which would look like a passing test if caught wrongly.
    """
    ts = pd.Timestamp(value)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read only the given columns via pyarrow's column projection.

    train.parquet on the large EB-NeRD bundle is ~8GB / 20M+ rows with
    click_history/candidates/labels list-columns — a plain `pd.read_parquet`
    materializes those as Python objects and reliably OOMs. None of these
    tests need those columns, so skip reading them entirely.
    """
    return pq.read_table(path, columns=columns).to_pandas()


def load_split_columns(proc_dir: Path, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = _read_columns(proc_dir / "train.parquet", columns)
    val = _read_columns(proc_dir / "val.parquet", columns)
    test = _read_columns(proc_dir / "test.parquet", columns)
    return train, val, test


@pytest.mark.parametrize("dataset,proc_dir", [
    ("mind",   Path("data/mind/processed")),
    ("ebnerd", Path("data/ebnerd/processed")),
])
def test_temporal_ordering(dataset, proc_dir):
    """Val and test timestamps must be strictly after train timestamps."""
    if not (proc_dir / "train.parquet").exists():
        pytest.skip(f"Processed data not found for {dataset}. Run build_pipeline.py first.")

    train, val, test = load_split_columns(proc_dir, ["timestamp"])

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

    train, val, test = load_split_columns(proc_dir, ["impression_id"])

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

    train, val, test = load_split_columns(proc_dir, ["timestamp"])

    train_ts = pd.to_datetime(train["timestamp"], utc=True)
    val_ts   = pd.to_datetime(val["timestamp"],   utc=True)
    test_ts  = pd.to_datetime(test["timestamp"],  utc=True)

    assert train_ts.max() < val_ts.min(),  f"[{dataset}] Train and val time ranges overlap!"
    assert val_ts.max()   < test_ts.min(), f"[{dataset}] Val and test time ranges overlap!"


# ── Behaviour-window boundary (Q9: "no future-click leakage") ─────────────────

_HISTORY_SAMPLE_ROWS = 20_000


@pytest.mark.parametrize("dataset,proc_dir", [
    ("mind",   Path("data/mind/processed")),
    ("ebnerd", Path("data/ebnerd/processed")),
])
def test_history_articles_predate_impression(dataset, proc_dir):
    """No article in a click_history was published after its own impression.

    Requires article publication timestamps, which only EB-NeRD provides;
    MIND is covered by test_history_is_monotone_per_user instead.
    """
    if not (proc_dir / "val.parquet").exists():
        pytest.skip(f"Processed data not found for {dataset}.")

    articles = _read_columns(proc_dir / "articles.parquet", ["article_id", "published_time"])
    published = {
        aid: _as_naive(ts)
        for aid, ts in zip(articles["article_id"], articles["published_time"])
        if pd.notna(ts)
    }
    if not published:
        pytest.skip(f"{dataset} has no article publication timestamps (MIND's news.tsv omits them).")

    batch = next(
        pq.ParquetFile(proc_dir / "val.parquet").iter_batches(
            columns=["timestamp", "click_history"], batch_size=_HISTORY_SAMPLE_ROWS
        )
    ).to_pandas()

    violations = checked = 0
    for impression_ts, history in zip(batch["timestamp"], batch["click_history"]):
        shown_at = _as_naive(impression_ts)
        for article_id in history:
            published_at = published.get(article_id)
            if published_at is None:
                continue
            checked += 1
            if published_at > shown_at:
                violations += 1

    assert checked > 0, f"[{dataset}] no history articles had a publication time to check"
    assert violations == 0, (
        f"[{dataset}] LEAKAGE: {violations:,} of {checked:,} click_history articles were "
        f"published AFTER the impression that lists them"
    )


@pytest.mark.parametrize("dataset,proc_dir", [
    ("mind",   Path("data/mind/processed")),
    ("ebnerd", Path("data/ebnerd/processed")),
])
def test_history_is_monotone_per_user(dataset, proc_dir):
    """For one user, click_history may only grow as time advances.

    A future click leaking backwards into an earlier impression would make
    that impression's history contain an article absent from a later one.
    """
    if not (proc_dir / "val.parquet").exists():
        pytest.skip(f"Processed data not found for {dataset}.")

    batch = next(
        pq.ParquetFile(proc_dir / "val.parquet").iter_batches(
            columns=["user_id", "timestamp", "click_history"], batch_size=_HISTORY_SAMPLE_ROWS
        )
    ).to_pandas().sort_values(["user_id", "timestamp"])

    shrank = users_checked = 0
    for _, group in itertools.islice(batch.groupby("user_id"), _HISTORY_SAMPLE_ROWS):
        if len(group) < 2:
            continue
        users_checked += 1
        previous = None
        for history in group["click_history"]:
            current = set(history)
            if previous is not None and not previous.issubset(current):
                shrank += 1
                break
            previous = current

    assert users_checked > 0, f"[{dataset}] no users had multiple impressions to compare"
    assert shrank == 0, (
        f"[{dataset}] LEAKAGE: {shrank:,} of {users_checked:,} users had a click_history that "
        f"lost articles over time — an earlier impression saw clicks a later one did not"
    )
