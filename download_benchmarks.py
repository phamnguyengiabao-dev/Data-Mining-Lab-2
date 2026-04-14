#!/usr/bin/env python
"""Download benchmark datasets required for the Apriori report."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path("data/benchmark")
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
    )
}

DATASETS = {
    "chess.txt": {
        "urls": [
            "https://www.philippe-fournier-viger.com/spmf/publicdatasets/chess.txt",
            "https://raw.githubusercontent.com/webmasterdy/SPMF/master/ca/pfviger/spmf/datastructures/dataset/chess.txt",
        ],
        "label": "Chess",
    },
    "mushroom.txt": {
        "urls": [
            "https://www.philippe-fournier-viger.com/spmf/publicdatasets/mushrooms.txt",
            "https://raw.githubusercontent.com/webmasterdy/SPMF/master/ca/pfviger/spmf/datastructures/dataset/mushrooms.txt",
        ],
        "label": "Mushroom",
    },
    "retail.txt": {
        "urls": [
            "https://www.philippe-fournier-viger.com/spmf/publicdatasets/retail.txt",
            "https://raw.githubusercontent.com/webmasterdy/SPMF/master/ca/pfviger/spmf/datastructures/dataset/retail.txt",
        ],
        "label": "Retail",
    },
    "T10I4D100K.txt": {
        "urls": [
            "https://www.philippe-fournier-viger.com/spmf/publicdatasets/T10I4D100K.txt",
            "https://raw.githubusercontent.com/webmasterdy/SPMF/master/ca/pfviger/spmf/datastructures/dataset/T10I4D100K.txt",
        ],
        "label": "T10I4D100K",
    },
    "accidents.txt": {
        "urls": [
            "https://www.philippe-fournier-viger.com/spmf/publicdatasets/accidents.txt",
            "https://raw.githubusercontent.com/webmasterdy/SPMF/master/ca/pfviger/spmf/datastructures/dataset/accidents.txt",
        ],
        "label": "Accidents (optional large dataset)",
    },
}


def download_to(filepath: Path, url: str) -> None:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request) as response, filepath.open("wb") as output:
        output.write(response.read())


def main() -> None:
    print("Downloading benchmark datasets into data/benchmark/")
    print("")

    success = 0
    failed = []

    for filename, info in DATASETS.items():
        filepath = BENCHMARK_DIR / filename
        print(f"- {info['label']}")

        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  already exists: {filename} ({size:,} bytes)")
            success += 1
            continue

        downloaded = False
        for url in info["urls"]:
            try:
                download_to(filepath, url)
                size = filepath.stat().st_size
                print(f"  downloaded: {filename} ({size:,} bytes)")
                downloaded = True
                success += 1
                break
            except Exception as exc:
                if filepath.exists():
                    filepath.unlink()
                print(f"  failed from {url}: {exc}")

        if not downloaded:
            failed.append(filename)

    print("")
    print(f"Completed: {success} dataset(s) ready.")
    if failed:
        print("Failed:")
        for filename in failed:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
