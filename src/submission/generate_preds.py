"""
src/submission/generate_preds.py — Generate Codabench prediction files (Q5).

MIND format   : one line per impression → "ImpressionID [N1-rank N2-rank ...]"
EB-NeRD format: parquet with columns impression_id, article_id, score

Usage
-----
    python -m src.submission.generate_preds --dataset mind --retriever bm25
    python -m src.submission.generate_preds --dataset ebnerd --retriever semantic
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("predictions")


def rank_impression(candidates: list[str], scores: list[float]) -> list[tuple[str, int]]:
    """Return (article_id, rank) sorted by descending score."""
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [(aid, i + 1) for i, (aid, _) in enumerate(ranked)]


def generate_mind_predictions(retriever, test_df: pd.DataFrame, output_path: Path):
    """Generate MIND-format prediction file."""
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    lines = []

    log.info(f"  Generating MIND predictions for {len(test_df):,} impressions...")
    for _, row in test_df.iterrows():
        candidates = row["candidates"]

        if hasattr(retriever, "retrieve"):
            # Use retriever to score/rank candidates
            retrieved = retriever.retrieve(row.get("click_history", []), k=len(candidates) + 50)
            # Score each candidate by its position in retrieved list (lower pos = higher score)
            retrieved_pos = {aid: i for i, aid in enumerate(retrieved)}
            scores = [1.0 / (retrieved_pos.get(aid, len(retrieved)) + 1) for aid in candidates]
        else:
            # Fallback: uniform random scores
            scores = np.random.rand(len(candidates)).tolist()

        ranked = rank_impression(candidates, scores)
        rank_str = " ".join(f"{aid}-{rank}" for aid, rank in ranked)
        lines.append(f"{row['impression_id']} [{rank_str}]")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    log.info(f"  Saved MIND predictions → {output_path}")


def generate_ebnerd_predictions(retriever, test_df: pd.DataFrame, output_path: Path):
    """Generate EB-NeRD-format prediction parquet."""
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    rows = []

    log.info(f"  Generating EB-NeRD predictions for {len(test_df):,} impressions...")
    for _, row in test_df.iterrows():
        candidates = row["candidates"]

        if hasattr(retriever, "retrieve"):
            retrieved = retriever.retrieve(row.get("click_history", []), k=len(candidates) + 50)
            retrieved_pos = {aid: i for i, aid in enumerate(retrieved)}
            scores = [1.0 / (retrieved_pos.get(aid, len(retrieved)) + 1) for aid in candidates]
        else:
            scores = np.random.rand(len(candidates)).tolist()

        for aid, score in zip(candidates, scores):
            rows.append({
                "impression_id": row["impression_id"],
                "article_id": aid,
                "score": score,
            })

    pred_df = pd.DataFrame(rows)
    pred_df.to_parquet(output_path, index=False)
    log.info(f"  Saved EB-NeRD predictions → {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--retriever", choices=["bm25", "semantic"], default="bm25")
    parser.add_argument("--split",     default="test", help="Which split to predict on")
    args = parser.parse_args()

    from src.pipeline.feature_store import load_split

    # Load retriever
    if args.retriever == "bm25":
        from src.retrieval.bm25 import BM25Retriever
        retriever = BM25Retriever(args.dataset)
        retriever.build()
    else:
        from src.retrieval.semantic import SemanticRetriever
        retriever = SemanticRetriever(args.dataset)
        retriever.build()

    test_df = load_split(args.dataset, args.split)

    if args.dataset == "mind":
        out = PREDICTIONS_DIR / f"mind_{args.retriever}_predictions.txt"
        generate_mind_predictions(retriever, test_df, out)
    else:
        out = PREDICTIONS_DIR / f"ebnerd_{args.retriever}_predictions.parquet"
        generate_ebnerd_predictions(retriever, test_df, out)
