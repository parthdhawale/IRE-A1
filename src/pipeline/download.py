"""
src/pipeline/download.py — Download raw datasets for MIND and EB-NeRD.

Q1 requirement: downloads raw data files for both MIND-small and EB-NeRD demo/small.

MIND is downloaded directly from Microsoft Azure blob storage (no login needed).
EB-NeRD is downloaded from AWS S3 using Python requests (no wget needed).
"""

import logging
import sys
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

MIND_RAW_DIR = Path("data/mind/raw")
EBNERD_RAW_DIR = Path("data/ebnerd/raw")

# ── MIND — Public mirror (no HuggingFace login required) ──────────────────────
# Hosted by the Microsoft Recommenders team
MIND_URLS = {
    "small_train": "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_train.zip",
    "small_dev":   "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_dev.zip",
    "large_test":  "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDlarge_test.zip",
    "large_train": "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDlarge_train.zip",
    "large_dev":   "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDlarge_dev.zip",
}

# ── EB-NeRD S3 URLs ───────────────────────────────────────────────────────────
EBNERD_URLS = {
    "demo":  "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
    "small": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
    "large": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip",
    "testset": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip",
}


def _download_file(url: str, dest_dir: Path) -> Path:
    """Download a file via Python requests with a tqdm progress bar. No wget needed."""
    import requests
    from tqdm import tqdm

    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0]  # strip query params if any
    dest = dest_dir / filename

    if dest.exists():
        log.info(f"  Already exists, skipping: {dest}")
        return dest

    log.info(f"  Downloading {filename} ...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=filename, ncols=80
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))
    log.info(f"  Saved → {dest}")
    return dest


def _unzip(zip_path: Path, dest_dir: Path):
    """Unzip a file using Python's built-in zipfile (no system zip needed)."""
    log.info(f"  Unzipping {zip_path.name} → {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    log.info(f"  Unzipped OK.")


def download_ebnerd(bundles=("demo", "small")):
    """Download and unzip EB-NeRD bundles from AWS S3."""
    log.info("Downloading EB-NeRD dataset...")
    EBNERD_RAW_DIR.mkdir(parents=True, exist_ok=True)

    for bundle in bundles:
        if bundle not in EBNERD_URLS:
            raise ValueError(f"Unknown EB-NeRD bundle '{bundle}'. Choose from {list(EBNERD_URLS)}")
        zip_path = _download_file(EBNERD_URLS[bundle], EBNERD_RAW_DIR)
        _unzip(zip_path, EBNERD_RAW_DIR)

    log.info(f"  EB-NeRD raw data saved to: {EBNERD_RAW_DIR}")


def download_mind(bundles=("small_train", "small_dev")):
    """
    Download MIND from the public Microsoft Recommenders mirror on HuggingFace.
    No login required.
    """
    log.info("Downloading MIND from public mirror (no login required)...")
    MIND_RAW_DIR.mkdir(parents=True, exist_ok=True)

    for key in bundles:
        url = MIND_URLS[key]
        zip_path = _download_file(url, MIND_RAW_DIR)
        _unzip(zip_path, MIND_RAW_DIR)

    log.info(f"  MIND raw data saved to: {MIND_RAW_DIR}")


def download_all(dataset: str = "both", include_large: bool = False):
    """Entry point called by build_pipeline.py."""
    ebnerd_bundles = ["demo", "small"]
    mind_bundles = ["small_train", "small_dev"]

    if include_large:
        ebnerd_bundles.extend(["large", "testset"])
        mind_bundles.extend(["large_train", "large_dev", "large_test"])

    if dataset in ("both", "ebnerd"):
        download_ebnerd(bundles=tuple(ebnerd_bundles))
    if dataset in ("both", "mind"):
        download_mind(bundles=tuple(mind_bundles))

