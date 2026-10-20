"""US benchmark step 1: data-center campuses and buildings from OpenStreetMap, five hubs.

Run (repo root):  python -m dct.us.osm_sites [--overwrite]
Output (data/seed_us/, ODbL, every row data_source=osm):
  campuses.csv, phases.csv, sources.csv (+ empty milestones/claims/observations),
  footprints.gpkg        one dissolved outline per campus (column campus_id, hub, area_m2, n_buildings)
  phase_footprints.gpkg  one outline per building/phase (column phase_id, campus_id, area_m2)
  outputs/us/osm_sites_summary.csv   counts per hub

Campus building rule: OSM features within merge_distance_m of each other (touching/overlapping or
close) form one campus. A feature that contains another feature (e.g. a campus outline around
buildings) is treated as the campus outline, not as a building. Remaining features are phases.
Point-only features and footprints below min_footprint_m2 are dropped and counted.
Status is 'under_construction' if any member has a construction=* tag, else 'operational'
(a placeholder; stage labels come later from anchors).
"""
from __future__ import annotations

import argparse
import time
from datetime import date

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.ops import transform as shp_transform, unary_union
from shapely.strtree import STRtree

from dct.common import REPO_ROOT, load_us_config, out_dir, seed_path, utm_epsg, write_seed
from dct.fetch.osm_footprints import _geometry, _post
from dct.schema import MODELS

SOURCE_ID = "S_OSM"


def _query(bbox: list[float]) -> str:
    w, s, e, n = bbox
    tags = ['"telecom"="data_center"', '"building"="data_center"', '"construction"="data_center"',
            '"industrial"="data_center"']
    parts = "".join(f"nwr[{t}]({s},{w},{n},{e});" for t in tags)
    return f"[out:json][timeout:120];({parts});out tags geom;"


def fetch_features(cfg: dict, hub_key: str) -> list[dict]:
    data = _post(cfg, _query(cfg["clusters"][hub_key]["bbox"]))
    feats = []
    for el in data.get("elements", []):
        geom = _geometry(el)
        if geom is None or geom.is_empty:
            continue
        tags = el.get("tags", {})
        feats.append({
            "osm_id": f"{el['type'][0]}{el['id']}", "name": tags.get("name", ""),
            "operator": tags.get("operator", ""),
            "under_construction": any(k in tags for k in ("construction",)) or tags.get("building") == "construction",
            "geometry": geom,
        })
    return feats


def _components(geoms_utm: list, merge_m: float) -> list[list[int]]:
    """Union-find over geometries whose merge_m-buffers intersect."""
    n = len(geoms_utm)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buffered = [g.buffer(merge_m / 2.0) for g in geoms_utm]
    tree = STRtree(buffered)
    for i, g in enumerate(buffered):
        for j in tree.query(g):
            j = int(j)
            if j != i and g.intersects(buffered[j]):
                parent[find(i)] = find(j)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def build_hub(cfg: dict, hub_key: str, feats: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    us = cfg["us"]
    hub = us["hubs"][hub_key]
    short = hub_key.replace("us_", "").upper()
    bbox = hub["bbox"]
    lon0, lat0 = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    utm = utm_epsg(lon0, lat0)
    fwd = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
    inv = Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform

    stats = {"hub": hub_key, "features": len(feats), "points_dropped": 0, "small_dropped": 0}
    polys = []
    for f in feats:
        g = f["geometry"]
        if g.geom_type not in ("Polygon", "MultiPolygon"):
            stats["points_dropped"] += 1
            continue
        g_utm = shp_transform(fwd, g)
        if not g_utm.is_valid:
            g_utm = g_utm.buffer(0)
        if g_utm.area < us["sites"]["min_footprint_m2"]:
            stats["small_dropped"] += 1
            continue
        polys.append({**f, "utm": g_utm})

    campuses, phases, camp_geoms, phase_geoms = [], [], [], []
    comps = _components([p["utm"] for p in polys], us["sites"]["merge_distance_m"])
    for comp in comps:
        members = [polys[i] for i in comp]
        # leaf geometries = buildings; containers (outlines that hold another member) are not phases
        reps = [m["utm"].representative_point() for m in members]
        leaves = []
        for a, m in enumerate(members):
            contains_other = any(b != a and m["utm"].contains(reps[b]) and m["utm"].area > members[b]["utm"].area
                                 for b in range(len(members)))
            if not contains_other:
                leaves.append(m)
        if not leaves:
            leaves = members
        campus_utm = unary_union([m["utm"] for m in members])
        # stable id: the smallest OSM id among the members, so re-runs and bbox tweaks do not renumber campuses
        anchor = min(members, key=lambda m: (int(m["osm_id"][1:]), m["osm_id"][0]))
        campus_id = f"US_{short}_{anchor['osm_id'].upper()}"
        leaves.sort(key=lambda m: (int(m["osm_id"][1:]), m["osm_id"][0]))
        rp = campus_utm.representative_point()
        lon, lat = inv(rp.x, rp.y)
        names = [m["name"] for m in members if m["name"]]
        ops = [m["operator"] for m in members if m["operator"]]
        name = min(set(names), key=lambda n: (-names.count(n), n)) if names else f"Data center near {lat:.3f}, {lon:.3f}"
        operator = min(set(ops), key=lambda n: (-ops.count(n), n)) if ops else ""
        campuses.append({
            "campus_id": campus_id, "name": name, "program": "", "owner": "", "operator": operator, "tenant": "",
            "country": "US", "state": hub.get("state", ""), "locality": hub["name"], "cluster": hub_key,
            "lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "location_precision": "exact", "location_method": "osm",
            "area_ha": "", "status": "under_construction" if any(m["under_construction"] for m in members) else "operational",
            "it_mw_total": "", "grid_mw": "", "investment_text": "", "osm_id": ";".join(m["osm_id"] for m in members),
            "latest_news": "", "latest_news_date": "", "risk_note": "", "main_source_id": SOURCE_ID,
            "notes": "", "data_source": us["sites"]["data_source"], "licence": us["sites"]["licence"],
        })
        camp_geoms.append({"campus_id": campus_id, "hub": hub_key, "area_m2": round(campus_utm.area, 1),
                           "n_buildings": len(leaves), "geometry": shp_transform(inv, campus_utm)})
        for b, m in enumerate(leaves, start=1):
            pid = f"{campus_id}_B{b}"
            phases.append({"phase_id": pid, "campus_id": campus_id, "name": m["name"] or f"Building {b}",
                           "it_mw": "", "status": campuses[-1]["status"], "notes": m["osm_id"]})
            phase_geoms.append({"phase_id": pid, "campus_id": campus_id, "area_m2": round(m["utm"].area, 1),
                                "geometry": shp_transform(inv, m["utm"])})
    stats.update(campuses=len(campuses), phases=len(phases))
    return campuses, phases, camp_geoms, phase_geoms, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="replace an existing data/seed_us")
    args = ap.parse_args()
    cfg = load_us_config()
    seed = REPO_ROOT / cfg["paths"]["seed_dir"]
    seed.mkdir(parents=True, exist_ok=True)
    existing = seed_path(cfg, "campuses")
    if existing.exists() and len(pd.read_csv(existing)) and not args.overwrite:
        raise SystemExit(f"{existing} already has rows. Use --overwrite to rebuild (auto-generated OSM rows only).")

    all_c, all_p, all_cg, all_pg, summary, failed = [], [], [], [], [], []
    for hub_key in cfg["us"]["hubs"]:
        print(f"{hub_key}: querying Overpass ...")
        try:
            feats = fetch_features(cfg, hub_key)
        except RuntimeError as exc:
            print(f"  FAILED: {exc}")
            failed.append(hub_key)
            continue
        c, p, cg, pg, st = build_hub(cfg, hub_key, feats)
        print(f"  {st['features']} features -> {st['campuses']} campuses, {st['phases']} buildings "
              f"(dropped: {st['points_dropped']} points, {st['small_dropped']} small)")
        all_c += c; all_p += p; all_cg += cg; all_pg += pg; summary.append(st)
        time.sleep(3)  # be polite to the public mirrors

    if failed:
        raise SystemExit(f"Overpass failed for {failed}; nothing written (a missing hub would silently shrink the "
                         "benchmark). Re-run, ideally later when the mirrors are less busy.")
    cols = {name: list(model.model_fields) for name, model in MODELS.items()}
    write_seed(cfg, "campuses", pd.DataFrame(all_c, columns=cols["campuses"]))
    write_seed(cfg, "phases", pd.DataFrame(all_p, columns=cols["phases"]))
    write_seed(cfg, "sources", pd.DataFrame([{
        "source_id": SOURCE_ID, "url": "https://www.openstreetmap.org/", "publisher": "OpenStreetMap contributors",
        "title": "OSM data_center features via Overpass (ODbL)", "pub_date": "", "accessed": date.today().isoformat(),
        "tier": cfg["us"]["sites"]["source_tier"]}], columns=cols["sources"]))
    for name in ("milestones", "claims", "observations"):
        write_seed(cfg, name, pd.DataFrame(columns=cols[name]))
    if all_cg:
        gpd.GeoDataFrame(all_cg, crs="EPSG:4326").to_file(seed / "footprints.gpkg", driver="GPKG")
        gpd.GeoDataFrame(all_pg, crs="EPSG:4326").to_file(seed / "phase_footprints.gpkg", driver="GPKG")
    o = out_dir(cfg)
    o.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(o / "osm_sites_summary.csv", index=False)
    print(f"\n{len(all_c)} campuses, {len(all_p)} phases written to {seed}")
    print("Next: python -m dct.build_db --us")


if __name__ == "__main__":
    main()
