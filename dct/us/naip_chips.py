"""US benchmark step 3: NAIP chips per campus per flight year (LABELS ONLY, never model input).

Run (repo root):  python -m dct.us.naip_chips [--hub us_nova] [--limit 20]
Needs data/seed_us (python -m dct.us.osm_sites) and footprints.

Sources
  Planetary Computer 'naip' (2010-2023, anonymous signing, public domain): STAC search, windowed COG read.
  USDA county mosaics for 2024-2026 (https://nrcs.app.box.com/v/naip): download by hand into
  naip.usda_local_dir; the year is read from the file path (a 4-digit 20xx). Readable formats depend on
  your GDAL build (GeoTIFF fine; MrSID is usually NOT readable by conda GDAL). Dates of mosaics are
  approximate (July of that year, flagged date_approx=true). Several county files can cover one campus: only the
  file with the most valid (non-black) pixels is kept per year, and chips that are mostly black are dropped.

Output: data/raw/us/naip/<campus_id>/<year>_<item>.tif  (uint8, bands red,green,blue,nir, EPSG of the
source, 1.0 m, n x n chips of 256 px centred on the campus) + .json with date, item and flags.
Writes are atomic (json first, tif renamed into place last), so a tif that exists is complete.
The same campus-year is skipped if its tif exists. Manifest: data/raw/us/naip_manifest.csv (date = ISO flight date).
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date

import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.ops import transform as shp_transform

from dct.common import (MANIFEST_COLUMNS, REPO_ROOT, load_us_config, manifest_row, raw_dir, read_seed,
                        require_footprints, utm_epsg)


def campus_layout(cfg: dict, footprint) -> tuple[float, float, int, float]:
    """(centre_lon, centre_lat, n_chips_per_side, size_m) for a campus footprint (EPSG:4326)."""
    ncfg = cfg["us"]["naip"]
    c = footprint.centroid
    utm = utm_epsg(c.x, c.y)
    fp_utm = shp_transform(Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform, footprint)
    minx, miny, maxx, maxy = fp_utm.bounds
    need = max(maxx - minx, maxy - miny) + 2 * ncfg["margin_m"]
    chip_m = ncfg["chip_px"] * ncfg["target_gsd_m"]
    n = 1 if need <= chip_m else min(int(ncfg["max_chips_side"]), int(np.ceil(need / chip_m)))
    return float(c.x), float(c.y), n, n * chip_m


def read_chip(src: rasterio.DatasetReader, lon: float, lat: float, size_m: float, size_px: int,
              resampling: Resampling = Resampling.bilinear):
    """Read a size_m square centred on (lon, lat), resampled to size_px x size_px.
    Returns (array, transform, crs). Use Resampling.nearest for categorical bands such as the S2 SCL."""
    cx, cy = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform(lon, lat)
    half = size_m / 2.0
    left, bottom, right, top = cx - half, cy - half, cx + half, cy + half
    win = from_bounds(left, bottom, right, top, transform=src.transform)
    data = src.read(window=win, out_shape=(src.count, size_px, size_px), resampling=resampling,
                    boundless=True, fill_value=0)
    return data, transform_from_bounds(left, bottom, right, top, size_px, size_px), src.crs


def valid_fraction(data: np.ndarray) -> float:
    """Share of pixels that are not black in the colour bands."""
    return float((data[:3].sum(0) > 0).mean())


def _done(tif) -> bool:
    return tif.exists() and tif.with_suffix(".json").exists()


def _save(out_tif, data: np.ndarray, transform, crs, meta: dict) -> None:
    nir_missing = data.shape[0] < 4
    if nir_missing:                                   # RGB-only flight: pad a zero NIR band
        data = np.concatenate([data[:3], np.zeros_like(data[:1])], axis=0)
    meta["nir_missing"] = nir_missing
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    part = out_tif.with_name(out_tif.name + ".part")
    with rasterio.open(part, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=4,
                       dtype="uint8", crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(data[:4].astype("uint8"))
    out_tif.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    part.replace(out_tif)                              # last: the tif only appears once everything is written


def _pc_items_by_year(client: Client, ncfg: dict, lon: float, lat: float) -> dict[int, list]:
    """All items per flight year that contain the point (neighbouring quads overlap); the caller keeps the best chip."""
    search = client.search(collections=[ncfg["pc_collection"]], intersects={"type": "Point", "coordinates": [lon, lat]})
    by_year: dict[int, list] = {}
    for item in search.items():
        year = int(item.properties.get("naip:year") or item.datetime.year)
        if year >= ncfg["min_year"]:
            by_year.setdefault(year, []).append(item)
    return by_year


def _index_local(cfg: dict) -> list[dict]:
    root = REPO_ROOT / cfg["us"]["naip"]["usda_local_dir"]
    idx = []
    if not root.exists():
        return idx
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".tif", ".tiff", ".jp2", ".img"):
            continue
        # NAIP file names carry the FLIGHT date then the production date (m_3807708_ne_18_060_20231005_20240212.tif):
        # prefer the first full YYYYMMDD token; fall back to the last 4-digit year in the path.
        m8 = re.search(r"(?<!\d)(20[12]\d)(0[1-9]|1[0-2])([0-2]\d|3[01])(?!\d)", path.stem)
        if m8:
            year = int(m8.group(1))
        else:
            m = re.findall(r"(?<!\d)(20[12]\d)(?!\d)", str(path.relative_to(root)))
            if not m:
                print(f"  skip {path.name}: no year in its path")
                continue
            year = int(m[-1])
        if year < cfg["us"]["naip"]["min_year"]:
            continue
        try:
            with rasterio.open(path) as src:
                b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        except Exception as exc:                      # unreadable format (e.g. MrSID)
            print(f"  skip {path.name}: cannot open ({exc})")
            continue
        idx.append({"path": path, "year": year, "bounds": b})
    return idx


def run(cfg: dict, hub: str | None = None, limit: int | None = None) -> list[dict]:
    ncfg = cfg["us"]["naip"]
    min_valid = ncfg["min_valid_fraction"]
    client = Client.open(ncfg["pc_stac"])
    footprints = require_footprints(cfg)
    campuses = read_seed(cfg, "campuses")
    if hub:
        campuses = campuses[campuses["cluster"] == hub]
    if limit:
        campuses = campuses.head(limit)
    local = _index_local(cfg)
    print(f"{len(campuses)} campuses; {len(local)} local USDA file(s) indexed")

    rows = []
    for rec in campuses.to_dict("records"):
        cid = rec["campus_id"]
        match = footprints[footprints["campus_id"] == cid]
        if match.empty:
            continue
        lon, lat, n, size_m = campus_layout(cfg, match.geometry.union_all())
        size_px = n * ncfg["chip_px"]
        base = {"campus_id": cid, "n_chips_side": n, "chip_px": ncfg["chip_px"], "gsd_m": ncfg["target_gsd_m"],
                "centre_lonlat": [lon, lat]}

        # --- Planetary Computer years
        try:
            items = _pc_items_by_year(client, ncfg, lon, lat)
        except Exception as exc:
            print(f"{cid}: STAC search failed ({exc})")
            items = {}
        for year, cands in sorted(items.items()):
            out = raw_dir(cfg, "naip", cid, f"{year}_pc.tif")
            if _done(out):
                meta = json.loads(out.with_suffix(".json").read_text())
                rows.append(manifest_row(cid, "naip", meta["date"], meta["item_id"], out, "cached", f"year={year}"))
                continue
            best, errors = None, []
            for item in cands:                              # overlapping quads: keep the chip with the most valid pixels
                try:
                    with rasterio.open(pc.sign(item).assets["image"].href) as src:
                        data, transform, crs = read_chip(src, lon, lat, size_m, size_px)
                except Exception as exc:
                    errors.append(str(exc)[:120])
                    continue
                vf = valid_fraction(data)
                if best is None or vf > best[0]:
                    best = (vf, item, data, transform, crs)
            if best is None:
                rows.append(manifest_row(cid, "naip", f"{year}-07-01", cands[0].id, None, "error", f"year={year}; {errors[:1]}"))
                continue
            vf, item, data, transform, crs = best
            day = item.datetime.date().isoformat()
            if vf < min_valid:
                rows.append(manifest_row(cid, "naip", day, item.id, None, "gap", f"year={year}; valid {vf:.2f}"))
                continue
            _save(out, data, transform, crs, {**base, "year": year, "date": day, "date_approx": False,
                                              "item_id": item.id, "source": "planetary-computer",
                                              "gsd_native": item.properties.get("gsd"), "valid_fraction": vf})
            rows.append(manifest_row(cid, "naip", day, item.id, out, "ok", f"year={year}"))

        # --- local USDA mosaics (2024-2026): best valid file per year
        best: dict[int, tuple] = {}
        for entry in local:
            w, s, e, n_ = entry["bounds"]
            if not (w <= lon <= e and s <= lat <= n_) or entry["year"] in items:
                continue
            out = raw_dir(cfg, "naip", cid, f"{entry['year']}_usda.tif")
            if _done(out):
                best[entry["year"]] = ("cached", out, entry)
                continue
            try:
                with rasterio.open(entry["path"]) as src:
                    data, transform, crs = read_chip(src, lon, lat, size_m, size_px)
            except Exception as exc:
                rows.append(manifest_row(cid, "naip", f"{entry['year']}-07-01", entry["path"].stem, None, "error", str(exc)[:160]))
                continue
            vf = valid_fraction(data)
            if vf >= min_valid and (entry["year"] not in best or best[entry["year"]][0] == "new" and vf > best[entry["year"]][1]):
                best[entry["year"]] = ("new", vf, entry, data, transform, crs)
        for year, b in sorted(best.items()):
            day = f"{year}-07-01"
            if b[0] == "cached":
                rows.append(manifest_row(cid, "naip", day, b[2]["path"].stem, b[1], "cached", f"year={year}; date_approx"))
                continue
            _, vf, entry, data, transform, crs = b
            out = raw_dir(cfg, "naip", cid, f"{year}_usda.tif")
            _save(out, data, transform, crs, {**base, "year": year, "date": day, "date_approx": True,
                                              "item_id": entry["path"].stem, "source": "usda-local", "valid_fraction": vf})
            rows.append(manifest_row(cid, "naip", day, entry["path"].stem, out, "ok", f"year={year}; date_approx"))
        n_ok = sum(1 for r in rows if r["campus_id"] == cid and r["status"] in ("ok", "cached"))
        print(f"{cid}: {n_ok} chip file(s)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_us_config()
    rows = run(cfg, args.hub, args.limit)
    path = raw_dir(cfg, "naip_manifest.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    old = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=MANIFEST_COLUMNS)
    pd.concat([old, new]).drop_duplicates(subset=["campus_id", "sensor", "date", "item_id"], keep="last").to_csv(path, index=False)
    print(f"manifest: {path} ({new['status'].value_counts().to_dict()}) on {date.today()}")


if __name__ == "__main__":
    main()
