"""Benchmark splits: leave-one-hub-out folds + a temporal hold-out.

Run:  python -m dct.benchmark.splits
Writes data/processed/us/splits.json:
  {"hub_folds": {"us_nova": {"train": [campus ids], "test": [campus ids]}, ...},
   "temporal": {"cutoff_month": "YYYY-MM", "train_months_before": "...", "test_months_from": "..."}}
evaluate.py builds the same folds internally from the hub of each campus; this file is for the release
and so others can reproduce the exact folds.
"""
from __future__ import annotations

import json

import pandas as pd

from dct.benchmark.dataset import month_keys
from dct.common import load_us_config, proc_dir, read_seed


def build(cfg: dict) -> dict:
    lab = pd.read_csv(proc_dir(cfg, "monthly_labels.csv"), usecols=["campus_id"])
    labelled = set(lab["campus_id"])
    c = read_seed(cfg, "campuses")
    c = c[c["campus_id"].isin(labelled)]
    folds = {}
    for hub in sorted(c["cluster"].unique()):
        folds[hub] = {"train": sorted(c.loc[c["cluster"] != hub, "campus_id"]),
                      "test": sorted(c.loc[c["cluster"] == hub, "campus_id"])}
    months = month_keys(cfg)
    hold = cfg["us"]["models"]["eval"]["temporal_holdout_months"]
    return {"hub_folds": folds,
            "temporal": {"holdout_months": hold, "cutoff_month": months[-hold],
                         "train_months_before": months[-hold], "test_months_from": months[-hold]}}


def main() -> None:
    cfg = load_us_config()
    s = build(cfg)
    p = proc_dir(cfg, "splits.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=1))
    print(f"{len(s['hub_folds'])} hub folds, temporal cutoff {s['temporal']['cutoff_month']} -> {p}")


if __name__ == "__main__":
    main()
