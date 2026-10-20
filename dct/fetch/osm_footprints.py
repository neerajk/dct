"""Candidate data-center outlines from OpenStreetMap (Overpass), one GeoPackage per cluster.

Run: python -m dct.fetch.osm_footprints
Output: data/raw/osm/<cluster>.gpkg  (columns: osm_id, name, operator, geometry)
These are CANDIDATES. Copy the ones that match your campuses into data/seed/footprints.gpkg
(column campus_id) in QGIS. India clusters are mostly unmapped; trace those by hand.

Overpass returns 406 without a User-Agent and 502 when busy, so this retries across mirrors.
"""
from __future__ import annotations

import time

import geopandas as gpd
import requests
from shapely.geometry import Point, Polygon
from shapely.ops import polygonize, unary_union
from shapely.geometry import LineString

from dct.common import load_config, raw_dir


def _query(bbox: list[float]) -> str:
    w, s, e, n = bbox
    return (
        f'[out:json][timeout:90];('
        f'nwr["telecom"="data_center"]({s},{w},{n},{e});'
        f'nwr["building"="data_center"]({s},{w},{n},{e});'
        f');out tags geom;'
    )


def _post(cfg: dict, query: str) -> dict:
    ocfg = cfg["osm"]
    headers = {"User-Agent": ocfg["user_agent"], "Accept": "application/json"}
    last = None
    for attempt in range(3):
        for url in ocfg["endpoints"]:
            try:
                r = requests.post(url, data={"data": query}, headers=headers, timeout=ocfg["timeout_s"])
                if r.status_code == 200:
                    js = r.json()
                    remark = str(js.get("remark", ""))
                    # Overpass reports timeouts / out-of-memory as HTTP 200 with a remark and partial or empty elements
                    if any(w in remark.lower() for w in ("runtime error", "timed out", "out of memory", "too much")):
                        last = f"{url}: {remark[:120]}"
                        continue
                    return js
                last = f"{url}: HTTP {r.status_code}"
            except (requests.RequestException, ValueError) as exc:
                last = f"{url}: {exc}"
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Overpass failed on all mirrors ({last})")


def _geometry(el: dict):
    if el["type"] == "node":
        return Point(el["lon"], el["lat"])
    if el["type"] == "way":
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            return Polygon(pts)
        return Point(pts[0]) if pts else None
    if el["type"] == "relation":
        lines = []
        for m in el.get("members", []):
            if m.get("role") == "outer" and m.get("geometry"):
                lines.append(LineString([(p["lon"], p["lat"]) for p in m["geometry"]]))
        polys = list(polygonize(unary_union(lines))) if lines else []
        return unary_union(polys) if polys else None
    return None


def fetch_cluster(cfg: dict, name: str, bbox: list[float]) -> gpd.GeoDataFrame:
    data = _post(cfg, _query(bbox))
    rows = []
    for el in data.get("elements", []):
        geom = _geometry(el)
        if geom is None:
            continue
        tags = el.get("tags", {})
        rows.append({
            "osm_id": f"{el['type'][0]}{el['id']}", "name": tags.get("name", ""),
            "operator": tags.get("operator", ""), "geometry": geom,
        })
    return gpd.GeoDataFrame(rows, columns=["osm_id", "name", "operator", "geometry"], geometry="geometry", crs="EPSG:4326")


def main() -> None:
    cfg = load_config()
    out_dir = raw_dir(cfg, "osm")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, c in cfg["clusters"].items():
        out = out_dir / f"{name}.gpkg"
        if out.exists():
            print(f"{name}: cached ({out})")
            continue
        try:
            gdf = fetch_cluster(cfg, name, c["bbox"])
        except RuntimeError as exc:
            print(f"{name}: FAILED - {exc}")
            continue
        if len(gdf):
            gdf.to_file(out, driver="GPKG")
        print(f"{name}: {len(gdf)} OSM feature(s)")
        time.sleep(3)  # be polite to the public mirrors


if __name__ == "__main__":
    main()
