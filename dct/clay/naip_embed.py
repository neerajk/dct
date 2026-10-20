"""US benchmark step 4a: Clay v1.5 embeddings of every NAIP campus-year (input to the teacher).

Run (ML env):  python -m dct.clay.naip_embed [--hub us_nova] [--limit 20]
Reads  data/raw/us/naip/<campus>/<year>_*.tif (+ .json) from dct.us.naip_chips
Writes data/processed/us/naip_emb/<campus>.npz/.csv  with arrays cls, inside, ring (n_rows, 1024)
        and index columns campus_id, year, date, item_id, n_inside_patches, hub.
A campus with an existing store is skipped (resume) unless a NAIP tif is missing from it (new years) or --refresh. Each tif is split into n x n chips of 256 px;
per-chip embeddings are averaged weighted by the number of inside-footprint patches.
NAIP is LABEL-SIDE only: these embeddings feed the teacher, never the deployed tracker.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import rasterio

from dct.clay.encoder import ClayEncoder, encode_latlon, encode_time
from dct.clay.pooling import embed_chips, patch_masks
from dct.clay.store import save_emb
from dct.common import load_us_config, proc_dir, raw_dir, read_seed, require_footprints
from dct.runtime import add_device_arg, apply_device_arg

PLATFORM = "naip"


def split_chips(data: np.ndarray, transform, chip_px: int):
    """data (4, S, S) -> list of (chip (4,chip_px,chip_px), chip_transform)."""
    n = data.shape[1] // chip_px
    out = []
    for r in range(n):
        for c in range(n):
            chip = data[:, r * chip_px:(r + 1) * chip_px, c * chip_px:(c + 1) * chip_px]
            out.append((chip, transform * transform.translation(c * chip_px, r * chip_px)))
    return out


def embed_campus(cfg: dict, enc: ClayEncoder, campus_id: str, footprint, hub: str, refresh: bool = False) -> int:
    ccfg, ncfg = cfg["us"]["clay"], cfg["us"]["naip"]
    prefix = proc_dir(cfg, "naip_emb", campus_id)
    folder = raw_dir(cfg, "naip", campus_id)
    files = sorted(folder.glob("*.tif")) if folder.exists() else []
    if prefix.with_suffix(".npz").exists() and not refresh:
        # stale if tifs exist that are not in the stored index (e.g. USDA 2024-26 added after the first run)
        have = set(pd.read_csv(prefix.with_suffix(".csv"))["item_id"].astype(str))
        want = {json.loads(t.with_suffix(".json").read_text())["item_id"] for t in files if t.with_suffix(".json").exists()}
        if want <= have:
            return -1
    rows, cls_l, in_l, ring_l = [], [], [], []
    for tif in files:
        meta = json.loads(tif.with_suffix(".json").read_text())
        with rasterio.open(tif) as src:
            data, transform, crs = src.read(), src.transform, src.crs
        chips = split_chips(data, transform, ncfg["chip_px"])
        lon, lat = meta["centre_lonlat"]
        d = pd.Timestamp(meta["date"])
        arr, ins, rng, n_in = [], [], [], []
        for chip, ctr in chips:
            i, r = patch_masks(footprint, ctr, crs, ncfg["chip_px"], enc.patch_size, ncfg["target_gsd_m"], ccfg["ring_m"])
            arr.append(chip.astype("float32")); ins.append(i); rng.append(r); n_in.append(int(i.sum()))
        k = len(arr)
        emb = embed_chips(enc, np.stack(arr), PLATFORM, [encode_time(d)] * k, [encode_latlon(lat, lon)] * k,
                          np.stack(ins), np.stack(rng), enc.batch_size)
        w = np.array(n_in, dtype="float32")
        w = w / w.sum()
        cls_l.append((emb["cls"] * w[:, None]).sum(0)); in_l.append((emb["inside"] * w[:, None]).sum(0))
        ring_l.append((emb["ring"] * w[:, None]).sum(0))
        rows.append({"campus_id": campus_id, "year": meta["year"], "date": meta["date"], "item_id": meta["item_id"],
                     "n_inside_patches": int(sum(n_in)), "hub": hub, "date_approx": bool(meta.get("date_approx", False))})
    if not rows:
        return 0
    save_emb(prefix, pd.DataFrame(rows), {"cls": np.stack(cls_l), "inside": np.stack(in_l), "ring": np.stack(ring_l)})
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refresh", action="store_true", help="re-embed campuses that already have a store")
    add_device_arg(ap)
    args = ap.parse_args()
    apply_device_arg(args)
    cfg = load_us_config()
    footprints = require_footprints(cfg)
    campuses = read_seed(cfg, "campuses")
    if args.hub:
        campuses = campuses[campuses["cluster"] == args.hub]
    if args.limit:
        campuses = campuses.head(args.limit)
    enc = ClayEncoder(cfg)
    print(f"Clay v1.5 on {enc.dev} (dtype {enc.amp_dtype or 'float32'}, batch {enc.batch_size}); {len(campuses)} campuses")
    for rec in campuses.to_dict("records"):
        cid = rec["campus_id"]
        match = footprints[footprints["campus_id"] == cid]
        if match.empty:
            continue
        n = embed_campus(cfg, enc, cid, match.geometry.union_all(), rec["cluster"], args.refresh)
        print(f"{cid}: {'cached' if n < 0 else str(n) + ' campus-year embedding(s)'}")


if __name__ == "__main__":
    main()
