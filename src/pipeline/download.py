"""
src/pipeline/download.py — Download raw datasets for MIND and EB-NeRD.

Q1 requirement: downloads raw data files for both MIND-small and EB-NeRD demo/small.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

MIND_RAW_DIR = Path("data/mind/raw")
EBNERD_RAW_DIR = Path("data/ebnerd/raw")

# ── EB-NeRD S3 URLs ───────────────────────────────────────────────────────────
EBNERD_URLS = {
    "demo": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "small": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
    # Uncomment when ready for Codabench submission (several GB):
    # "large":    "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip",
    # "articles_large": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/articles_large_only.zip",
    # "testset":  "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip",
    # Optional pre-trained embeddings:
    # "word2vec": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip",
    # "bert":     "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip",
}

# ── MIND HuggingFace repo ─────────────────────────────────────────────────────
MIND_HF_REPO = "yjw1029/MIND"


def _wget(url: str, dest_dir: Path):
    """Download a file via requests with a progress bar."""
    import requests
    from tqdm import tqdm

    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest = dest_dir / filename

    if dest.exists():
        log.info(f"  Already exists, skipping: {dest}")
        return dest

    log.info(f"  Downloading {filename} ...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def _unzip(zip_path: Path, dest_dir: Path):
    import zipfile

    log.info(f"  Unzipping {zip_path.name} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def download_ebnerd(bundles=("demo",)):
    """Download and unzip EB-NeRD bundles."""
    log.info("Downloading EB-NeRD dataset...")
    EBNERD_RAW_DIR.mkdir(parents=True, exist_ok=True)

    for bundle in bundles:
        if bundle not in EBNERD_URLS:
            raise ValueError(f"Unknown EB-NeRD bundle '{bundle}'. Choose from {list(EBNERD_URLS)}")
        zip_path = _wget(EBNERD_URLS[bundle], EBNERD_RAW_DIR)
        _unzip(zip_path, EBNERD_RAW_DIR)

    log.info(f"  EB-NeRD raw data saved to: {EBNERD_RAW_DIR}")


def download_mind():
    """Download MIND-small dataset from HuggingFace."""
    log.info("Downloading MIND dataset from HuggingFace...")
    MIND_RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=MIND_HF_REPO,
            repo_type="dataset",
            local_dir=str(MIND_RAW_DIR),
        )
        log.info(f"  MIND raw data saved to: {MIND_RAW_DIR}")
    except ImportError:
        log.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        log.error(f"  Failed to download MIND: {e}")
        raise


def download_all(dataset: str = "both"):
    """Entry point called by build_pipeline.py."""
    if dataset in ("both", "ebnerd"):
        download_ebnerd(bundles=("demo",))  # use 'small' for full training run
    if dataset in ("both", "mind"):
        download_mind()
