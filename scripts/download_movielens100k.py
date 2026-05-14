#!/usr/bin/env python3
"""Download and extract MovieLens 100K into data/ml-100k.

This script uses the public GroupLens URL. It is optional because the repository
also includes a tiny MovieLens-style sample dataset for offline testing.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="Directory that will contain ml-100k")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "ml-100k.zip"
    print(f"Downloading {URL} -> {zip_path}")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"Extracted to {out_dir / 'ml-100k'}")
    print("You can now run: python -m cinematch.cli all --data data/ml-100k --out outputs/ml-100k --fast")


if __name__ == "__main__":
    main()
