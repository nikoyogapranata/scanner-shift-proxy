"""Get the compact dataset onto a Colab (or Kaggle) machine, the fast way.

    from ssp import fetch
    os.environ["MNMS_ROOT"] = fetch.ensure_data("/content/drive/MyDrive/ssp_data")

First time: downloads the Kaggle copy, builds the compact copy, stores it on
Google Drive as one file (mnms_compact.tar), and deletes the large download.
Every later session: copies that one file from Drive and unpacks it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from . import compact

KAGGLE_DATASET = "tailength/m-and-ms-dataset"
TAR_NAME = "mnms_compact.tar"


def ensure_data(drive_dir: str, local_dir: str = "/content/mnms_compact") -> str:
    drive, local = Path(drive_dir), Path(local_dir)
    tar = drive / TAR_NAME
    if (local / compact.META_NAME).exists():
        print(f"Compact data already unpacked at {local}")
        return str(local)
    if tar.exists():
        print(f"Unpacking {tar} ...")
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(tar), "-C", str(local.parent)], check=True)
        print(f"Ready: {local}")
        return str(local)

    print("No compact copy on Drive yet. Building it once (about 15 to 30 minutes).")
    import kagglehub

    src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    compact.build(src, local)
    drive.mkdir(parents=True, exist_ok=True)
    print(f"Saving {tar} to Google Drive ...")
    subprocess.run(["tar", "-cf", str(tar), "-C", str(local.parent), local.name], check=True)
    print(f"Saved ({tar.stat().st_size / 1e9:.2f} GB).")
    cache = src
    while cache.name != "datasets" and cache.parent != cache:
        cache = cache.parent
    shutil.rmtree(cache, ignore_errors=True)          # free the ~70 GB the original download used
    print("Deleted the full download to free disk space.")
    return str(local)
