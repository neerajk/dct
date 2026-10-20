"""Run every synchronous fetcher in order and write data/raw/manifest.csv.

Run: python -m dct.fetch.run_all [--only sentinel2 landsat_lst aerial_nl aerial_hessen]

Sentinel-1 (HyP3) is asynchronous and has its own commands - see sentinel1_hyp3.py.
LISS-IV is deliberately not included yet. Esri World Imagery is view-only and never fetched.
Every fetcher is skip-if-cached, so an interrupted run is safe to repeat.
"""
from __future__ import annotations

import argparse

import pandas as pd

from dct.common import MANIFEST_COLUMNS, load_config, raw_dir
from dct.fetch import aerial_hessen, aerial_nl, landsat_lst, sentinel2

FETCHERS = {
    "aerial_nl": aerial_nl.run,
    "aerial_hessen": aerial_hessen.run,
    "sentinel2": sentinel2.run,
    "landsat_lst": landsat_lst.run,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(FETCHERS), default=list(FETCHERS))
    args = ap.parse_args()
    cfg = load_config()

    path = raw_dir(cfg, "manifest.csv")
    old = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)
    new_rows = []
    for name in args.only:
        print(f"\n=== {name} ===")
        try:
            new_rows += FETCHERS[name](cfg)
        except Exception as exc:  # keep going; the manifest records what did finish
            print(f"{name} FAILED: {exc}")
    new = pd.DataFrame(new_rows, columns=MANIFEST_COLUMNS)
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates(
        subset=["campus_id", "sensor", "date", "item_id"], keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    print(f"\nmanifest: {path} ({len(merged)} rows; "
          f"{merged['status'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
