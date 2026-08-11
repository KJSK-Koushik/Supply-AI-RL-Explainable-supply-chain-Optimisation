"""Download the raw datasets into data/raw/.

Raw data is not committed: it is ~50 MB, and redistribution terms are not ours
to grant. This script fetches what can be fetched automatically.

    python scripts/fetch_data.py

UCI Online Retail II (the primary calibration source) downloads directly.
The Kaggle file needs an account, so it is fetched via the Kaggle CLI if
credentials are present, and otherwise reported with manual instructions.
It is only needed to reproduce the dataset-screening section of the report --
the simulator does not depend on it.
"""

from __future__ import annotations

import io
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
UCI_FILE = "online_retail_II.xlsx"

KAGGLE_DATASET = "anirudhchauhan/retail-store-inventory-forecasting-dataset"
KAGGLE_FILE = "retail_store_inventory.csv"


def fetch_uci() -> bool:
    target = RAW / UCI_FILE
    if target.exists():
        print(f"[uci ] already present: {target.name}")
        return True

    print(f"[uci ] downloading {UCI_URL} (~45 MB) ...")
    try:
        with urllib.request.urlopen(UCI_URL, timeout=120) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(RAW)
        print(f"[uci ] extracted {target.name}")
        return True
    except Exception as exc:
        print(f"[uci ] FAILED: {exc}")
        print(f"[uci ] download manually from {UCI_URL} and unzip into {RAW}")
        return False


def fetch_kaggle() -> bool:
    target = RAW / KAGGLE_FILE
    if target.exists():
        print(f"[kagl] already present: {target.name}")
        return True

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        print("[kagl] no ~/.kaggle/kaggle.json -- skipping automatic download")
        print(f"[kagl] optional: download {KAGGLE_DATASET} manually into {RAW}")
        print("[kagl] only needed to reproduce the dataset-screening section")
        return False

    print(f"[kagl] downloading {KAGGLE_DATASET} ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW), "--unzip"],
            check=True,
        )
        return target.exists()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[kagl] FAILED: {exc}  (pip install kaggle)")
        return False


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    ok_uci = fetch_uci()
    fetch_kaggle()  # optional, never fatal

    if not ok_uci:
        print("\nUCI is required -- the pipeline cannot run without it.")
        return 1
    print("\nReady. Next: python -m src.data.run_pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
