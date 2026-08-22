"""
src/submission/generate_preds.py — Generate Codabench prediction files (Q5).

Both competitions require the SAME text format (verified directly against
each competition's own "Submission Guidelines" page — MIND:
codabench.org/competitions/13967, EB-NeRD: codabench.org/competitions/2469):

    ImpressionID [Rank-of-candidate-1,Rank-of-candidate-2,...,Rank-of-candidate-N]

One line per impression, ranks are integers 1..N (1 = best), comma-separated
with no spaces, and — critically — listed in the SAME ORDER the candidates
appeared in the original impression, not sorted by rank and not paired with
article IDs. Both require a zip containing exactly one file and nothing
else (no folder, no __MACOSX) — but the required filename differs:
MIND wants "prediction.txt" (singular), EB-NeRD wants "predictions.txt"
(plural). Getting either of these wrong causes a failed or misparsed
submission, so this was checked against each competition's real
instructions rather than assumed.

By default this scores the official, unlabeled Codabench test bundle
(impressions_competition_test.parquet, produced by preprocess_all() from
MINDlarge_test / ebnerd_testset — only present if the pipeline was run with
--large) — this is the file that must actually be submitted to the
leaderboards. Pass --split test to instead score our own internal temporal
test split (data/{dataset}/processed/test.parquet), which is useful for a
prediction-format dry run but is NOT what Codabench expects, since it's
carved from labeled training data rather than the real held-out set.

Usage
-----
    python -m src.submission.generate_preds --dataset mind --retriever bm25
    python -m src.submission.generate_preds --dataset ebnerd --retriever semantic
"""

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

log = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("predictions")


def ranks_in_original_order(scores: list[float]) -> list[int]:
    """Rank each candidate 1..N (1 = highest score), returned in the SAME
    order `scores` was given in — not sorted. This is exactly what both
    competitions' format requires: a rank list positionally aligned to the
    impression's original candidate order, e.g. candidates
    [N125045, N87192, N73556, N20417] scoring N87192 highest and N125045
    lowest becomes [4,1,3,2] (verified against MIND's own worked example)."""
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-arr, kind="stable")  # indices, best score first
    ranks = np.empty(len(arr), dtype=int)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks.tolist()


def _score_row(retriever, row) -> list[float]:
    """Score one impression's own candidate list directly, via score_candidates()
    — NOT retriever.retrieve(), which searches the entire article catalog just
    to derive a proxy score for the handful of candidates actually shown. That
    pattern is exactly what made recall@K and the eval harness slow before they
    were fixed (see ai_usage_log.md); at competition-test scale (MIND: 2.37M
    impressions, EB-NeRD: 13.5M per the assignment) it would cost hours here
    too. score_candidates() computes only what's needed and gives the
    retriever's real relevance score instead of a rank-position proxy."""
    candidates = list(row["candidates"])
    if hasattr(retriever, "score_candidates"):
        return retriever.score_candidates(row.get("click_history", []), candidates)
    return np.random.rand(len(candidates)).tolist()  # fallback for an unscored retriever


_IMPRESSION_COLUMNS = ["impression_id", "click_history", "candidates"]


def iter_impression_batches(path: Path, batch_size: int = 100_000):
    """Stream an impressions parquet file in row-group batches instead of
    loading it fully into pandas.

    A plain pd.read_parquet() materializes every row's click_history and
    candidates lists as individual Python objects simultaneously — for a
    13.5M-row file (EB-NeRD's competition test set) that's tens of GB of
    object overhead, and it's what forced a hard OS-level shutdown of this
    whole session once already. Reading row-group batches keeps peak memory
    proportional to `batch_size`, not to the file's total row count. We also
    only request the 3 columns actually needed here (not user_id, timestamp,
    or labels — the labels are all-None anyway, this is the unlabeled
    competition set), which trims the per-batch cost further."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=_IMPRESSION_COLUMNS, batch_size=batch_size):
        yield batch.to_pandas()


def generate_prediction_lines(retriever, path: Path, total_rows: int) -> list[str]:
    """Score every impression's own candidates and format each as one
    'ImpressionID [rank1,rank2,...]' line — the shared format both
    competitions require (see module docstring)."""
    lines = []
    with tqdm(total=total_rows, desc="  Scoring", ncols=80) as bar:
        for batch_df in iter_impression_batches(path):
            for _, row in batch_df.iterrows():
                scores = _score_row(retriever, row)
                ranks = ranks_in_original_order(scores)
                rank_str = ",".join(str(r) for r in ranks)
                lines.append(f"{row['impression_id']} [{rank_str}]")
            bar.update(len(batch_df))
    return lines


def write_submission_zip(lines: list[str], zip_path: Path, inner_filename: str):
    """Zip `lines` as the single file `inner_filename` — nothing else in the
    archive, no folder wrapper, no __MACOSX — matching both competitions'
    'a valid zip submission should contain nothing but...' requirement."""
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_filename, "\n".join(lines))
    log.info(f"  Saved submission zip → {zip_path}  (contains {inner_filename})")


def generate_mind_predictions(retriever, path: Path, total_rows: int, output_path: Path):
    """Generate a MIND submission zip (inner file must be named prediction.txt)."""
    log.info(f"  Generating MIND predictions for {total_rows:,} impressions...")
    lines = generate_prediction_lines(retriever, path, total_rows)
    write_submission_zip(lines, output_path, "prediction.txt")


def generate_ebnerd_predictions(retriever, path: Path, total_rows: int, output_path: Path):
    """Generate an EB-NeRD submission zip (inner file must be named predictions.txt)."""
    log.info(f"  Generating EB-NeRD predictions for {total_rows:,} impressions...")
    lines = generate_prediction_lines(retriever, path, total_rows)
    write_submission_zip(lines, output_path, "predictions.txt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--retriever", choices=["bm25", "semantic"], default="bm25")
    parser.add_argument(
        "--split", default="competition_test",
        help="Which split to predict on. 'competition_test' (default) is the "
             "official unlabeled Codabench test set; 'test' is our own internal "
             "temporal test split (labeled, NOT for actual submission).",
    )
    args = parser.parse_args()

    from src.pipeline.feature_store import resolve_split_path

    # Load retriever
    if args.retriever == "bm25":
        from src.retrieval.bm25 import BM25Retriever
        retriever = BM25Retriever(args.dataset)
        retriever.build()
    else:
        from src.retrieval.semantic import SemanticRetriever
        retriever = SemanticRetriever(args.dataset)
        retriever.build()

    split_path = resolve_split_path(args.dataset, args.split)
    total_rows = pq.ParquetFile(split_path).metadata.num_rows

    if args.dataset == "mind":
        out = PREDICTIONS_DIR / f"mind_{args.retriever}_submission.zip"
        generate_mind_predictions(retriever, split_path, total_rows, out)
    else:
        out = PREDICTIONS_DIR / f"ebnerd_{args.retriever}_submission.zip"
        generate_ebnerd_predictions(retriever, split_path, total_rows, out)

    log.info(f"  Ready to upload: {out}")
