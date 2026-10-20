"""Sentinel-2 L2A chips per campus: the least-cloudy scene in each month, judged on the
Scene Classification Layer (SCL) INSIDE the campus AOI, not on the tile-wide cloud %.

Source: Earth Search STAC over the public AWS COGs (no auth; same pattern as potapulse
fetch/sentinel2.py). Saved raw (DN); meta.json records the processing baseline and whether
the +1000 DN BOA offset was already applied, so downstream code can correct reflectance.
Months without a clear enough scene are returned as 'gap' manifest rows (monsoon months in India).
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from dct.common import (campus_aoi_4326, campus_date_range, campuses_with_coords, load_config,
                        load_footprints, manifest_row, month_starts, raw_dir)


def _read_window(href: str, bbox_4326: tuple, boundless: bool = True):
    with rasterio.open(href) as src:
        native = transform_bounds("EPSG:4326", src.crs, *bbox_4326)
        win = from_bounds(*native, transform=src.transform)
        data = src.read(window=win, boundless=boundless, fill_value=src.nodata or 0)
        profile = src.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2],
                       transform=src.window_transform(win), driver="GTiff")
    return data, profile


def _bad_fraction(item, bbox_4326: tuple, bad_classes: list[int]) -> float:
    try:
        scl, _ = _read_window(item.assets["scl"].href, bbox_4326)
    except Exception:  # unreadable asset = unusable scene
        return 1.0
    return float(np.isin(scl[0], bad_classes).mean())


def _save_item(item, bbox_4326: tuple, assets: list[str], out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in assets:
        if a not in item.assets:
            continue
        data, profile = _read_window(item.assets[a].href, bbox_4326)
        with rasterio.open(out_dir / f"{a}.tif", "w", **profile) as dst:
            dst.write(data)
    props = item.properties
    (out_dir / "meta.json").write_text(json.dumps({
        "item_id": item.id, "datetime": props.get("datetime"),
        "eo:cloud_cover": props.get("eo:cloud_cover"),
        "s2:processing_baseline": props.get("s2:processing_baseline"),
        "earthsearch:boa_offset_applied": props.get("earthsearch:boa_offset_applied"),
        "bbox_4326": list(bbox_4326),
    }, indent=2))


def fetch_campus(cfg: dict, client: Client, campus: dict, footprints) -> list[dict]:
    s2 = cfg["sentinel2"]
    cid = campus["campus_id"]
    aoi = campus_aoi_4326(cfg, campus, footprints)
    bbox = tuple(aoi.bounds)
    start, end = campus_date_range(cfg, cid)

    search = client.search(
        collections=[s2["collection"]], bbox=bbox, datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": s2["max_scene_cloud"]}},
    )
    by_month: dict[str, list] = defaultdict(list)
    for item in search.items():
        by_month[item.datetime.strftime("%Y-%m")].append(item)

    rows = []
    for m in month_starts(start, end):
        key = m.strftime("%Y-%m")
        cands = sorted(by_month.get(key, []), key=lambda i: i.properties.get("eo:cloud_cover", 100))
        cands = cands[: s2["candidates_per_month"]]
        scored = [(_bad_fraction(i, bbox, s2["scl_bad_classes"]), i) for i in cands]
        scored.sort(key=lambda t: t[0])
        if not scored or scored[0][0] > s2["max_bad_fraction"]:
            best = f"{scored[0][0]:.2f}" if scored else "n/a"
            rows.append(manifest_row(cid, "sentinel2", key, "", None, "gap", f"best AOI bad fraction {best}"))
            continue
        frac, item = scored[0]
        out_dir = raw_dir(cfg, "s2", cid, f"{key}_{item.id}")
        if (out_dir / "meta.json").exists():
            rows.append(manifest_row(cid, "sentinel2", key, item.id, out_dir, "cached", f"bad {frac:.2f}"))
            continue
        _save_item(item, bbox, s2["assets"], out_dir)
        rows.append(manifest_row(cid, "sentinel2", key, item.id, out_dir, "ok", f"bad {frac:.2f}"))
        print(f"  {cid} {key} {item.id} bad={frac:.2f}")
    return rows


def run(cfg: dict) -> list[dict]:
    client = Client.open(cfg["sentinel2"]["stac_endpoint"])
    footprints = load_footprints(cfg)
    rows = []
    for campus in campuses_with_coords(cfg):
        print(f"Sentinel-2: {campus['campus_id']}")
        rows += fetch_campus(cfg, client, campus, footprints)
    return rows


if __name__ == "__main__":
    run(load_config())
