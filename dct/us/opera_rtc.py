"""US benchmark step 7: NASA OPERA RTC-S1 radar backscatter per campus-month (baseline features).

Run (ML env, needs a free Earthdata login in ~/.netrc):  python -m dct.us.opera_rtc [--hub us_nova] [--limit 20]
OPERA_L2_RTC-S1_V1 = terrain-corrected gamma0, 30 m, one burst per file (VV/VH COGs), near-global, 2016-04-14 on.
It replaces HyP3 RTC for backscatter (no credits, nothing to order).

Per campus: search granules over the campus bbox; keep only granules whose footprint polygon CONTAINS the campus
centroid; among those keep ONE burst id (the one with most acquisitions, so geometry is constant); take the first
acquisition of each month; read an 86 x 86 px (2.58 km) window of the VV and VH COGs over HTTPS (range requests, no
full download); months where less than 90% of the window is valid are skipped; mean linear power inside the footprint
and in the ring -> dB.
Output: data/processed/us/opera/<campus>.csv  month, date, granule, burst, vv_in, vv_ring, vv_diff, vh_in, vh_ring, vh_diff
(dB). Layover/shadow masks are NOT applied in v0 (OPERA ships them in the _mask.tif; a later improvement).
A campus with no OPERA granules gets an empty CSV (cached); a read error writes nothing, so it is retried.

UNTESTED ASSUMPTIONS (this module was written without being run):
  * earthaccess.get_fsspec_https_session() returns an authenticated fsspec filesystem and rasterio.open(url,
    opener=fs.open) (rasterio >= 1.4) then reads only the window by HTTP range requests. (Passing a file object from
    earthaccess.open() would instead load the whole file into memory.) If it fails, the error is printed per campus;
    fall back to downloading the two COGs per month with earthaccess.download.
  * the granule name layout OPERA_L2_RTC-S1_<burst>_<acq>_... (burst = field 3, acquisition = field 4) and the UMM
    footprint at umm.SpatialExtent.HorizontalSpatialDomain.Geometry.GPolygons[0].Boundary.Points.
  * values are linear power (converted to dB here). Check one file's units after the first run.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point, Polygon

from dct.common import load_us_config, proc_dir, read_seed, require_footprints
from dct.features.handcrafted import db, pixel_masks
from dct.us.naip_chips import read_chip

MIN_VALID = 0.9


def _retry(fn, what: str, tries: int = 3):
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{what} failed after {tries} tries: {last}")


def burst_and_day(granule) -> tuple[str, str]:
    name = granule["umm"]["GranuleUR"]
    parts = name.split("_")
    return parts[3], parts[4][:8]


def granule_polygon(granule):
    try:
        pts = granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
        return Polygon([(p["Longitude"], p["Latitude"]) for p in pts])
    except (KeyError, IndexError, TypeError):
        return None


def links(granule) -> dict[str, str]:
    out = {}
    for url in granule.data_links():
        for pol in ("VV", "VH"):
            if url.endswith(f"_{pol}.tif"):
                out[pol] = url
    return out


def process_campus(cfg: dict, rec: dict, footprint) -> str:
    import earthaccess
    ocfg, ccfg = cfg["us"]["opera"], cfg["us"]["clay"]
    cid = rec["campus_id"]
    out_csv = proc_dir(cfg, "opera", f"{cid}.csv")
    if out_csv.exists():
        return "cached"
    c = footprint.centroid
    lon, lat = float(c.x), float(c.y)
    pad = 0.02
    results = earthaccess.search_data(
        short_name=ocfg["short_name"], bounding_box=(lon - pad, lat - pad, lon + pad, lat + pad),
        temporal=(cfg["time"]["start_floor"], pd.Timestamp.today().strftime("%Y-%m-%d")), count=ocfg["count_per_search"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        pd.DataFrame(columns=["month"]).to_csv(out_csv, index=False)
        return "no OPERA granules (cached as empty)"

    fs = earthaccess.get_fsspec_https_session()          # authenticated fsspec filesystem (Earthdata login)
    pt = Point(lon, lat)
    parsed = [(g, *burst_and_day(g), granule_polygon(g)) for g in results]
    covering = [p for p in parsed if p[3] is not None and p[3].contains(pt)]
    note = ""
    if not covering:                                   # footprints unreadable: fall back, but say so
        covering, note = parsed, " (granule footprints not parsed: burst choice unchecked)"
    top = Counter(b for _, b, _, _ in covering).most_common(1)[0][0]
    by_month: dict[str, tuple] = {}
    for g, b, day, _ in sorted((p for p in covering if p[1] == top), key=lambda p: p[2]):
        by_month.setdefault(f"{day[:4]}-{day[4:6]}", (g, day))

    size_px = ocfg["chip_px"]
    size_m = size_px * ocfg["gsd_m"]
    rows, skipped = [], 0
    for month, (g, day) in sorted(by_month.items()):
        ln = links(g)
        if not {"VV", "VH"} <= ln.keys():
            continue
        vals = {}
        try:
            for pol in ("VV", "VH"):
                def _read(pol=pol):
                    with rasterio.open(ln[pol], opener=fs.open) as src:      # GDAL range requests: only the window is fetched
                        return read_chip(src, lon, lat, size_m, size_px)
                data, tr, cr = _retry(_read, f"{pol} read of {g['umm']['GranuleUR']}")
                vals[pol] = (data[0].astype("float32"), tr, cr)
        except Exception as exc:
            return f"FAILED reading {g['umm']['GranuleUR']}: {exc}"
        arrs = {pol: v[0] for pol, v in vals.items()}
        if any((np.isfinite(a) & (a > 0)).mean() < MIN_VALID for a in arrs.values()):
            skipped += 1                               # window only partly covered by this burst
            continue
        _, tr, cr = vals["VV"]
        ins, rng = pixel_masks(footprint, tr, cr, size_px, ccfg["ring_m"])
        row = {"month": month, "date": f"{day[:4]}-{day[4:6]}-{day[6:]}", "granule": g["umm"]["GranuleUR"], "burst": top}
        for pol in ("VV", "VH"):
            arr = arrs[pol]
            valid = np.isfinite(arr) & (arr > 0)
            mi, mr = ins & valid, rng & valid
            vin = float(db(arr[mi].mean())) if mi.any() else np.nan
            vring = float(db(arr[mr].mean())) if mr.any() else np.nan
            row.update({f"{pol.lower()}_in": vin, f"{pol.lower()}_ring": vring, f"{pol.lower()}_diff": vin - vring})
        rows.append(row)
    pd.DataFrame(rows, columns=["month", "date", "granule", "burst", "vv_in", "vv_ring", "vv_diff", "vh_in", "vh_ring", "vh_diff"]).to_csv(out_csv, index=False)
    warn = f"; WARNING all {skipped} month(s) only partly covered" if skipped and not rows else ""
    return f"{len(rows)} months (burst {top}, {skipped} partial skipped){note}{warn}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_us_config()
    footprints = require_footprints(cfg)
    import earthaccess
    earthaccess.login(strategy="netrc")
    campuses = read_seed(cfg, "campuses")
    if args.hub:
        campuses = campuses[campuses["cluster"] == args.hub]
    if args.limit:
        campuses = campuses.head(args.limit)
    for rec in campuses.to_dict("records"):
        match = footprints[footprints["campus_id"] == rec["campus_id"]]
        if match.empty:
            continue
        try:
            msg = process_campus(cfg, rec, match.geometry.union_all())
        except Exception as exc:
            msg = f"FAILED: {exc}"
        print(f"{rec['campus_id']}: {msg}")


if __name__ == "__main__":
    main()
