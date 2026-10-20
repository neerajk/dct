"""PDOK national aerial photos, every yearly edition, cropped to each NL campus.

Open data, CC BY 4.0 - credit "Kadaster / Beeldmateriaal Nederland".
Tile math and REST URL pattern come from potapulse fetch/pdok_aerial.py (verified live there);
the yearly layer names were read from GetCapabilities on 2026-10-19. 25 cm (ortho25) up to
2020, 8 cm (orthoHR) from 2021, rendered here at config aerial_nl.zoom_level.
One GeoTIFF (EPSG:28992, RGB uint8) per campus-year.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from io import BytesIO

import numpy as np
import rasterio
import requests
from PIL import Image
from pyproj import Transformer
from rasterio.transform import Affine

from dct.common import (campus_aoi_4326, campuses_with_coords, load_config, load_footprints,
                        manifest_row, raw_dir)

HTTP_TIMEOUT_S = 30


def _res_m(acfg: dict) -> float:
    return acfg["scale_denominator_level0"] * 0.00028 / (2 ** acfg["zoom_level"])


def _fetch_tile(acfg: dict, layer: str, col: int, row: int):
    url = acfg["tile_url"].format(layer=layer, level=acfg["zoom_level"], col=col, row=row)
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
        return np.array(Image.open(BytesIO(r.content)).convert("RGB"))
    except (requests.RequestException, OSError):
        return None


def fetch_crop(acfg: dict, layer: str, bounds_28992: tuple):
    """Mosaic of whole tiles covering the bounds. Returns (rgb HxWx3, Affine, n_missing_tiles)."""
    res, px = _res_m(acfg), acfg["tile_px"]
    tile_m = px * res
    minx, miny, maxx, maxy = bounds_28992
    c0, c1 = int((minx - acfg["topleft_x"]) // tile_m), int((maxx - acfg["topleft_x"]) // tile_m)
    r0, r1 = int((acfg["topleft_y"] - maxy) // tile_m), int((acfg["topleft_y"] - miny) // tile_m)
    n_tiles = (c1 - c0 + 1) * (r1 - r0 + 1)
    if n_tiles > acfg["max_tiles_per_crop"]:
        raise RuntimeError(f"{n_tiles} tiles > cap {acfg['max_tiles_per_crop']}: lower aoi.aerial_buffer_m or zoom_level")

    mosaic = np.zeros(((r1 - r0 + 1) * px, (c1 - c0 + 1) * px, 3), dtype="uint8")
    missing = [0]

    def work(rc):
        r, c = rc
        tile = _fetch_tile(acfg, layer, c, r)
        if tile is None:
            missing[0] += 1
            return
        mosaic[(r - r0) * px:(r - r0 + 1) * px, (c - c0) * px:(c - c0 + 1) * px] = tile

    tasks = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    with ThreadPoolExecutor(max_workers=min(acfg["max_parallel_fetches"], len(tasks))) as pool:
        list(pool.map(work, tasks))
    origin_x = acfg["topleft_x"] + c0 * tile_m
    origin_y = acfg["topleft_y"] - r0 * tile_m
    return mosaic, Affine(res, 0, origin_x, 0, -res, origin_y), missing[0]


def run(cfg: dict) -> list[dict]:
    acfg = cfg["aerial_nl"]
    footprints = load_footprints(cfg)
    to28992 = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
    rows = []
    for campus in campuses_with_coords(cfg):
        if campus["country"] != "NL":
            continue
        cid = campus["campus_id"]
        w, s, e, n = campus_aoi_4326(cfg, campus, footprints, kind="aerial").bounds
        (x0, y0), (x1, y1) = to28992.transform(w, s), to28992.transform(e, n)
        bounds = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        for year, layer in acfg["layers"].items():
            if year > date.today().year:
                continue
            out = raw_dir(cfg, "aerial_nl", cid, f"{year}_{layer}.tif")
            if out.exists():
                rows.append(manifest_row(cid, "aerial_nl", str(year), layer, out, "cached"))
                continue
            try:
                rgb, transform, missing = fetch_crop(acfg, layer, bounds)
            except RuntimeError as exc:
                rows.append(manifest_row(cid, "aerial_nl", str(year), layer, None, "error", str(exc)))
                continue
            total = (rgb.shape[0] // acfg["tile_px"]) * (rgb.shape[1] // acfg["tile_px"])
            if missing >= total:
                rows.append(manifest_row(cid, "aerial_nl", str(year), layer, None, "gap", "no tiles returned"))
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out, "w", driver="GTiff", height=rgb.shape[0], width=rgb.shape[1], count=3,
                               dtype="uint8", crs="EPSG:28992", transform=transform, compress="deflate") as dst:
                dst.write(np.moveaxis(rgb, 2, 0))
            rows.append(manifest_row(cid, "aerial_nl", str(year), layer, out, "ok", f"{missing}/{total} tiles missing"))
            print(f"  {cid} {year} {layer} ({missing}/{total} tiles missing)")
    return rows


if __name__ == "__main__":
    run(load_config())
