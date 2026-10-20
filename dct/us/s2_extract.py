"""US benchmark step 6: Sentinel-2 monthly chips per campus -> hand-made features + Clay v1.5 embeddings.

Run (ML env):  python -m dct.us.s2_extract [--hub us_nova] [--limit 20] [--no-clay] [--refresh]
For every campus and every COMPLETE month from 2017-01 (the still-open current month is skipped):
  1. STAC search (Earth Search sentinel-2-l2a); group scenes by month; for the clearest few (by tile cloud %) read
     the SCL over the campus chip and keep the scene with the smallest bad fraction over the campus FOOTPRINT + RING
     (then the whole chip). A month is a gap if the whole-chip bad fraction exceeds s2_clay.max_bad_fraction or the
     footprint+ring bad fraction exceeds max_bad_fraction_footprint. Gaps are recorded, never silently filled.
     A network/GDAL error is retried and then RAISED (the campus is reported FAILED and retried next run): it is never
     recorded as cloud.
  2. Read the 10 Clay bands as a 256 x 256 chip (2.56 km at 10 m) centred on the campus.
  3. Hand-made features (inside vs ring; cloud/shadow pixels masked out) and, unless --no-clay, Clay v1.5 CLS / inside /
     ring embeddings (these use every pixel; the footprint cloud gate keeps contamination small).
  No baseline-offset correction: items flagged earthsearch:boa_offset_applied=false were checked against live pixels
  and are already offset-free; flag and baseline are stored per month for audit only.
Outputs per campus:
  data/processed/us/s2/<campus>.csv        month, status (ok|gap), date, item_id, bad_frac, foot_bad_frac, baseline, offset_flag, features...
  data/processed/us/s2_emb/<campus>.npz/.csv   cls, inside, ring for the 'ok' months (written even when empty, so resume works)
Chips are not kept on disk (re-downloadable from item IDs): ~118 months x 500 campuses of 10-band chips would be tens of GB.
A campus is skipped if its outputs exist. After a new month has passed, re-run with --refresh to recompute campuses.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling

from dct.common import load_us_config, month_starts, proc_dir, read_seed, require_footprints
from dct.features.handcrafted import pixel_masks, s2_stats
from dct.runtime import add_device_arg, apply_device_arg
from dct.us.naip_chips import read_chip

PLATFORM = "sentinel-2-l2a"


def _retry(fn, what: str, tries: int = 3):
    """Call fn(); on any error retry with backoff, then RAISE (a persistent error is never recorded as cloud)."""
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{what} failed after {tries} tries: {last}")


def _read_asset(href: str, lon, lat, size_m, size_px, resampling=Resampling.bilinear):
    with rasterio.open(href) as src:
        return read_chip(src, lon, lat, size_m, size_px, resampling)


def _scl(item, lon, lat, size_m, size_px):
    """(scl array, transform, crs), or None if the item has no SCL asset."""
    if "scl" not in item.assets:
        return None
    d, tr, cr = _retry(lambda: _read_asset(item.assets["scl"].href, lon, lat, size_m, size_px, Resampling.nearest),
                       f"SCL read of {item.id}")
    return d[0], tr, cr


def select_month(args):
    """-> (month key, (chip_bad, foot_bad, item) of the best candidate, or None)."""
    key, cands, lon, lat, size_m, size_px, bad_classes, footprint, ring_m, max_chip, max_foot = args
    scored = []
    for item in cands:
        got = _scl(item, lon, lat, size_m, size_px)
        if got is None:
            continue
        scl, tr, cr = got
        bad = np.isin(scl, bad_classes)
        ins, rng = pixel_masks(footprint, tr, cr, size_px, ring_m)
        area = ins | rng
        foot = float(bad[area].mean()) if area.any() else 1.0
        scored.append((foot, float(bad.mean()), item))
    if not scored:
        return key, None
    # candidates that pass BOTH gates rank first (then by footprint, then chip, badness), so a month is a gap only if no scene passes
    scored.sort(key=lambda t: (not (t[1] <= max_chip and t[0] <= max_foot), t[0], t[1]))
    foot, chip, item = scored[0]
    return key, (chip, foot, item)


def _baseline(item) -> float:
    try:
        return float(str(item.properties.get("s2:processing_baseline", "0")))
    except ValueError:
        return 0.0


def process_campus(cfg: dict, client: Client, enc, rec: dict, footprint, refresh: bool = False) -> str:
    scfg, ccfg = cfg["us"]["s2_clay"], cfg["us"]["clay"]
    cid = rec["campus_id"]
    f_csv = proc_dir(cfg, "s2", f"{cid}.csv")
    e_prefix = proc_dir(cfg, "s2_emb", cid)
    if not refresh and f_csv.exists() and (enc is None or e_prefix.with_suffix(".npz").exists()):
        return "cached"

    stale_files = [f_csv] + ([e_prefix.with_suffix(".npz"), e_prefix.with_suffix(".csv")] if enc is not None else [])
    for stale in stale_files:                  # recomputing: never leave a half-old, half-new campus behind
        stale.unlink(missing_ok=True)

    c = footprint.centroid
    lon, lat = float(c.x), float(c.y)
    size_px = scfg["chip_px"]
    size_m = size_px * 10.0
    today = pd.Timestamp.today().normalize()
    this_month = today.replace(day=1)
    months = [m for m in month_starts(cfg["time"]["start_floor"], today.strftime("%Y-%m-%d")) if m < this_month]
    pad = 0.02
    search = client.search(
        collections=[cfg["sentinel2"]["collection"]], bbox=[lon - pad, lat - pad, lon + pad, lat + pad],
        datetime=f"{months[0].strftime('%Y-%m-%d')}/{this_month.strftime('%Y-%m-%d')}",
        query={"eo:cloud_cover": {"lt": cfg["sentinel2"]["max_scene_cloud"]}})
    by_month = defaultdict(list)
    for item in search.items():
        by_month[item.datetime.strftime("%Y-%m")].append(item)

    jobs = []
    for m in months:
        k = m.strftime("%Y-%m")
        cands = sorted(by_month.get(k, []), key=lambda i: i.properties.get("eo:cloud_cover", 100))
        jobs.append((k, cands[:cfg["sentinel2"]["candidates_per_month"]], lon, lat, size_m, size_px,
                     cfg["sentinel2"]["scl_bad_classes"], footprint, ccfg["ring_m"],
                     scfg["max_bad_fraction"], scfg["max_bad_fraction_footprint"]))
    with ThreadPoolExecutor(max_workers=scfg.get("io_workers_select", 8)) as pool:
        chosen = dict(pool.map(select_month, jobs))

    todo = [(k, v[2], v[0], v[1]) for k, v in chosen.items()
            if v is not None and v[0] <= scfg["max_bad_fraction"] and v[1] <= scfg["max_bad_fraction_footprint"]]

    def load(t):
        k, item, chip_bad, foot_bad = t
        arrs, tr, cr = [], None, None
        for b in scfg["bands"]:
            d, tr, cr = _retry(lambda b=b: _read_asset(item.assets[b].href, lon, lat, size_m, size_px),
                               f"{b} read of {item.id}")
            arrs.append(d[0])
        scl, _, _ = _scl(item, lon, lat, size_m, size_px)
        return k, item, chip_bad, foot_bad, np.stack(arrs).astype("float32"), tr, cr, np.isin(scl, cfg["sentinel2"]["scl_bad_classes"])

    with ThreadPoolExecutor(max_workers=scfg.get("io_workers_load", 6)) as pool:
        loaded = list(pool.map(load, todo))

    rows = {}
    for k, v in chosen.items():
        rows[k] = {"month": k, "status": "gap", "date": "", "item_id": "",
                   "bad_frac": v[0] if v else np.nan, "foot_bad_frac": v[1] if v else np.nan}
    chips, times, latlons, inside_p, ring_p, emb_keys = [], [], [], [], [], []
    for k, item, chip_bad, foot_bad, arr, tr, cr, bad in loaded:
        ins, rng = pixel_masks(footprint, tr, cr, size_px, ccfg["ring_m"])
        feats = s2_stats(arr, ins, rng, bad)
        rows[k].update({"status": "ok", "date": item.datetime.date().isoformat(), "item_id": item.id,
                        "bad_frac": chip_bad, "foot_bad_frac": foot_bad, "baseline": _baseline(item),
                        "offset_flag": str(item.properties.get("earthsearch:boa_offset_applied")), **feats})
        if enc is not None:
            from dct.clay.encoder import encode_latlon, encode_time
            from dct.clay.pooling import patch_masks
            pi, pr = patch_masks(footprint, tr, cr, size_px, enc.patch_size, 10.0, ccfg["ring_m"])
            chips.append(arr); inside_p.append(pi); ring_p.append(pr); emb_keys.append(k)
            times.append(encode_time(item.datetime, item.datetime.hour))
            latlons.append(encode_latlon(lat, lon))
    f_csv.parent.mkdir(parents=True, exist_ok=True)

    if enc is not None:
        from dct.clay.store import save_emb
        if chips:
            from dct.clay.pooling import embed_chips
            emb = embed_chips(enc, np.stack(chips), PLATFORM, times, latlons, np.stack(inside_p), np.stack(ring_p), enc.batch_size)
            index = pd.DataFrame({"campus_id": cid, "month": emb_keys})
        else:                                           # write an empty store so the resume check passes
            emb = {k: np.zeros((0, enc.dim), "float32") for k in ("cls", "inside", "ring")}
            index = pd.DataFrame({"campus_id": pd.Series(dtype=str), "month": pd.Series(dtype=str)})
        save_emb(e_prefix, index, emb)
    tmp = f_csv.with_name(f_csv.name + ".part")                 # the month table is written LAST (it is the resume marker)
    pd.DataFrame(list(rows.values())).to_csv(tmp, index=False)
    tmp.replace(f_csv)
    n_ok = sum(1 for r in rows.values() if r["status"] == "ok")
    return f"{n_ok}/{len(rows)} months ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-clay", action="store_true", help="features only (no torch needed)")
    ap.add_argument("--refresh", action="store_true", help="recompute campuses that already have outputs (new month, new settings)")
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
    enc = None
    if not args.no_clay:
        from dct.clay.encoder import ClayEncoder
        enc = ClayEncoder(cfg)
        print(f"Clay v1.5 on {enc.dev} (dtype {enc.amp_dtype or 'float32'}, batch {enc.batch_size})")
    client = Client.open(cfg["sentinel2"]["stac_endpoint"])
    for rec in campuses.to_dict("records"):
        match = footprints[footprints["campus_id"] == rec["campus_id"]]
        if match.empty:
            continue
        try:
            msg = process_campus(cfg, client, enc, rec, match.geometry.union_all(), args.refresh)
        except Exception as exc:                      # keep going; a failed campus writes nothing and is retried next run
            msg = f"FAILED: {exc}"
        print(f"{rec['campus_id']}: {msg}")


if __name__ == "__main__":
    main()
