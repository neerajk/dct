"""Tiny embedding store: one .npz (arrays) + one .csv (row index) per campus.

Chosen over parquet list-columns because it round-trips numpy arrays with no schema surprises and
makes resume trivial (a campus is done when its .npz exists; the .csv is written first and the .npz renamed into place last).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def save_emb(prefix: Path, index: pd.DataFrame, arrays: dict[str, np.ndarray]) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for k, a in arrays.items():
        if len(a) != len(index):
            raise ValueError(f"{k}: {len(a)} rows vs index {len(index)}")
    index.to_csv(prefix.with_suffix(".csv"), index=False)
    tmp = prefix.with_name(prefix.name + ".tmp.npz")             # a campus counts as done only when its .npz exists,
    np.savez_compressed(tmp, **{k: a.astype("float16") for k, a in arrays.items()})
    os.replace(tmp, prefix.with_suffix(".npz"))                 # so the .npz appears last, complete


def load_emb(prefix: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    idx = pd.read_csv(prefix.with_suffix(".csv"))
    z = np.load(prefix.with_suffix(".npz"))
    return idx, {k: z[k].astype("float32") for k in z.files}


def load_all(folder: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Concatenate every campus file in a folder (rows in sorted campus order)."""
    idxs, arrs = [], {}
    for csv in sorted(folder.glob("*.csv")):
        idx, a = load_emb(csv.with_suffix(""))
        if not len(idx):
            continue
        idxs.append(idx)
        for k, v in a.items():
            arrs.setdefault(k, []).append(v)
    if not idxs:
        return pd.DataFrame(), {}
    return pd.concat(idxs, ignore_index=True), {k: np.concatenate(v) for k, v in arrs.items()}
