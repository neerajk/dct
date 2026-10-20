"""Hessen DOP20 (20 cm RGB, licence dl-zero-de/2.0) cropped to each Hessen campus.

Only the current edition is served by this open WMS; a dated history for Frankfurt would need
another source (open question in the plan). WMS 1.1.1 GetMap in EPSG:25832, requested in
chunks of at most aerial_hessen.max_request_px per side, written as one GeoTIFF per campus
named by fetch date.
"""
from __future__ import annotations

from datetime import date
from io import BytesIO

import numpy as np
import rasterio
import requests
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin

from dct.common import (campus_aoi_4326, campuses_with_coords, load_config, load_footprints,
                        manifest_row, raw_dir)

HTTP_TIMEOUT_S = 60


def _getmap(hcfg: dict, bbox: tuple, w_px: int, h_px: int) -> np.ndarray:
    params = {
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap", "LAYERS": hcfg["layer"],
        "STYLES": "", "SRS": hcfg["crs"], "BBOX": ",".join(f"{v:.3f}" for v in bbox),
        "WIDTH": w_px, "HEIGHT": h_px, "FORMAT": "image/png",
    }
    r = requests.get(hcfg["wms_url"], params=params, timeout=HTTP_TIMEOUT_S)
    r.raise_for_status()
    if "image" not in r.headers.get("Content-Type", ""):
        raise RuntimeError(f"WMS returned {r.headers.get('Content-Type')}: {r.text[:200]}")
    return np.array(Image.open(BytesIO(r.content)).convert("RGB"))


def run(cfg: dict) -> list[dict]:
    hcfg = cfg["aerial_hessen"]
    res, max_px = hcfg["resolution_m"], hcfg["max_request_px"]
    footprints = load_footprints(cfg)
    to25832 = Transformer.from_crs("EPSG:4326", hcfg["crs"], always_xy=True)
    stamp = date.today().strftime("%Y-%m")
    rows = []
    for campus in campuses_with_coords(cfg):
        if campus["country"] != "DE":
            continue
        cid = campus["campus_id"]
        out = raw_dir(cfg, "aerial_hessen", cid, f"{stamp}_dop20.tif")
        if out.exists():
            rows.append(manifest_row(cid, "aerial_hessen", stamp, hcfg["layer"], out, "cached"))
            continue
        w, s, e, n = campus_aoi_4326(cfg, campus, footprints, kind="aerial").bounds
        (x0, y0), (x1, y1) = to25832.transform(w, s), to25832.transform(e, n)
        minx, miny, maxx, maxy = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
        W, H = int((maxx - minx) / res), int((maxy - miny) / res)
        mosaic = np.zeros((H, W, 3), dtype="uint8")
        try:
            for ty in range(0, H, max_px):
                for tx in range(0, W, max_px):
                    tw, th = min(max_px, W - tx), min(max_px, H - ty)
                    bb = (minx + tx * res, maxy - (ty + th) * res, minx + (tx + tw) * res, maxy - ty * res)
                    mosaic[ty:ty + th, tx:tx + tw] = _getmap(hcfg, bb, tw, th)
        except (requests.RequestException, RuntimeError) as exc:
            rows.append(manifest_row(cid, "aerial_hessen", stamp, hcfg["layer"], None, "error", str(exc)[:200]))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", driver="GTiff", height=H, width=W, count=3, dtype="uint8",
                           crs=hcfg["crs"], transform=from_origin(minx, maxy, res, res), compress="deflate") as dst:
            dst.write(np.moveaxis(mosaic, 2, 0))
        rows.append(manifest_row(cid, "aerial_hessen", stamp, hcfg["layer"], out, "ok"))
        print(f"  {cid} DOP20 {W}x{H}px")
    return rows


if __name__ == "__main__":
    run(load_config())
