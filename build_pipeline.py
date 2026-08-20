"""
build_pipeline.py — One-command entry point for the A1 data pipeline.

Usage:
    python build_pipeline.py [--dataset {mind,ebnerd,both}] [--skip-download]

Runs the full pipeline:
  1. Download raw data
  2. Preprocess into unified schema
  3. Temporal train/val/test split
  4. Build feature store
"""

import argparse
import logging
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def run_step(name: str, fn, *args, **kwargs):
    log.info(f"{'='*60}")
    log.info(f"  STEP: {name}")
    log.info(f"{'='*60}")
    t0 = time.time()
    fn(*args, **kwargs)
    log.info(f"  Done in {time.time() - t0:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Build the A1 data pipeline.")
    parser.add_argument(
        "--dataset",
        choices=["mind", "ebnerd", "both"],
        default="both",
        help="Which dataset(s) to process (default: both)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use if raw files already present)",
    )
    args = parser.parse_args()

    log.info("A1 Pipeline: starting end-to-end build")
    log.info(f"  Dataset  : {args.dataset}")
    log.info(f"  Skip DL  : {args.skip_download}\n")

    from src.pipeline.download import download_all
    from src.pipeline.preprocess import preprocess_all
    from src.pipeline.split import split_all
    from src.pipeline.feature_store import build_feature_store

    if not args.skip_download:
        run_step("Download raw data", download_all, dataset=args.dataset)

    run_step("Preprocess → unified schema", preprocess_all, dataset=args.dataset)
    run_step("Temporal train/val/test split", split_all, dataset=args.dataset)
    run_step("Build feature store", build_feature_store, dataset=args.dataset)

    log.info("Pipeline complete. Feature store is ready.")
    log.info("  MIND    → data/mind/processed/")
    log.info("  EB-NeRD → data/ebnerd/processed/")


if __name__ == "__main__":
    main()
