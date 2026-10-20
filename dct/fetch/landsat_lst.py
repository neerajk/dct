"""Landsat 8/9 Collection 2 Level-2 surface temperature (lwir11) chips per campus, one per month.

Source: Microsoft Planetary Computer STAC; assets are signed anonymously (verified for
sentinel-1-grd in potapulse; landsat-c2-l2 uses the same SAS signing - if signing ever needs a
key, planetary_computer will say so). Saved raw DN; meta.json carries scale/offset
(K = DN * scale + offset). A warm, persistent hot spot vs surroundings is the 'operational' cue.
Month choice = lowest scene cloud cover (a per-AOI QA_PIXEL test can replace it later).
"""
from __future__ import annotations

import json
from collections import defaultdict

import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from dct.common import (campus_aoi_4326, campus_date_range, campuses_with_coords, load_config,
                        load_footprints, manifest_row, month_starts, raw_dir)


def _scale_offset(item, asset_key: str, cfg: dict) -> tuple[float, float]:
    bands = item.assets[asset_key].extra_fields.get("raster:bands") or [{}]
    return (bands[0].get("scale", cfg["landsat"]["default_scale"]),
            bands[0].get("offset", cfg["landsat"]["default_offset"]))


def fetch_campus(cfg: dict, client: Client, campus: dict, footprints) -> list[dict]:
    lc = cfg["landsat"]
    cid = campus["campus_id"]
    bbox = tuple(campus_aoi_4326(cfg, campus, footprints).bounds)
    start, end = campus_date_range(cfg, cid)

    search = client.search(
        collections=[lc["collection"]], bbox=bbox, datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": lc["max_scene_cloud"]}, "platform": {"in": lc["platforms"]}},
    )
    by_month: dict[str, list] = defaultdict(list)
    for item in search.items():
        by_month[item.datetime.strftime("%Y-%m")].append(item)

    rows = []
    for m in month_starts(start, end):
        key = m.strftime("%Y-%m")
        cands = by_month.get(key, [])
        if not cands:
            rows.append(manifest_row(cid, "landsat_lst", key, "", None, "gap", "no scene under cloud limit"))
            continue
        item = min(cands, key=lambda i: i.properties.get("eo:cloud_cover", 100))
        out_dir = raw_dir(cfg, "landsat", cid, f"{key}_{item.id}")
        tif = out_dir / f"{lc['asset']}.tif"
        if tif.exists():
            rows.append(manifest_row(cid, "landsat_lst", key, item.id, tif, "cached"))
            continue
        signed = pc.sign(item)
        href = signed.assets[lc["asset"]].href
        with rasterio.open(href) as src:
            native = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*native, transform=src.transform)
            data = src.read(window=win, boundless=True, fill_value=0)
            profile = src.profile.copy()
            profile.update(height=data.shape[1], width=data.shape[2],
                           transform=src.window_transform(win), driver="GTiff")
        out_dir.mkdir(parents=True, exist_ok=True)
        with rasterio.open(tif, "w", **profile) as dst:
            dst.write(data)
        scale, offset = _scale_offset(item, lc["asset"], cfg)
        (out_dir / "meta.json").write_text(json.dumps({
            "item_id": item.id, "datetime": item.properties.get("datetime"),
            "eo:cloud_cover": item.properties.get("eo:cloud_cover"),
            "scale": scale, "offset": offset, "units": "kelvin = DN*scale + offset",
        }, indent=2))
        rows.append(manifest_row(cid, "landsat_lst", key, item.id, tif, "ok"))
        print(f"  {cid} {key} {item.id}")
    return rows


def run(cfg: dict) -> list[dict]:
    client = Client.open(cfg["landsat"]["stac_endpoint"])
    footprints = load_footprints(cfg)
    rows = []
    for campus in campuses_with_coords(cfg):
        print(f"Landsat LST: {campus['campus_id']}")
        rows += fetch_campus(cfg, client, campus, footprints)
    return rows


if __name__ == "__main__":
    run(load_config())
