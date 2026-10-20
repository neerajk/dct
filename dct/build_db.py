"""Validate the seed CSVs and build outputs/dct_projects.gpkg + outputs/gap_report.csv.

Run: python -m dct.build_db
Exit code 1 if there are validation errors (the GeoPackage is then not written).
"""
from __future__ import annotations

import argparse
import sys

import geopandas as gpd
import pandas as pd

from dct.common import REPO_ROOT, load_config, load_footprints, load_us_config
from dct.schema import validate

GAP_FIELDS = ["lat", "lon", "owner", "operator", "area_ha", "it_mw_total", "grid_mw",
              "main_source_id", "latest_news", "location_precision"]


def imagery_plan(cfg: dict, campuses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in campuses.to_dict("records"):
        s = cfg["sensors"].get(r["country"], {})
        rows.append({
            "campus_id": r["campus_id"],
            "sar": cfg["sensors"]["default_sar"],
            "optical": ";".join(s.get("optical", [])),
            "view_only": ";".join(s.get("view_only", [])),
        })
    return pd.DataFrame(rows)


def gap_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    campuses, phases, miles = tables["campuses"], tables["phases"], tables["milestones"]
    rows = []
    for r in campuses.to_dict("records"):
        missing = [f for f in GAP_FIELDS if not r[f]]
        pids = set(phases.loc[phases["campus_id"] == r["campus_id"], "phase_id"])
        if not pids:
            missing.append("phases")
        elif not miles["phase_id"].isin(pids).any():
            missing.append("milestones")
        rows.append({"campus_id": r["campus_id"], "name": r["name"],
                     "n_missing": len(missing), "missing": ";".join(missing)})
    return pd.DataFrame(rows)


def _attr_layer(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df.copy(), geometry=[None] * len(df))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", action="store_true", help="validate/build the US benchmark seed (data/seed_us)")
    ap.add_argument("--seed-dir", default=None, help="override the seed directory (relative to the repo root)")
    args = ap.parse_args()
    cfg = load_us_config() if args.us else load_config()
    if args.seed_dir:
        cfg["paths"]["seed_dir"] = args.seed_dir
    v = validate(cfg)
    out_dir = REPO_ROOT / cfg["paths"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for w in v.warnings:
        print(f"WARNING: {w}")
    for e in v.errors:
        print(f"ERROR: {e}")
    print(f"\n{len(v.errors)} error(s), {len(v.warnings)} warning(s)")

    t = v.tables
    gaps = gap_report(t)
    gaps.to_csv(out_dir / "gap_report.csv", index=False)

    c = t["campuses"]
    print(f"\nCampuses: {len(c)}  | by country: {c['country'].value_counts().to_dict()}")
    print(f"By status: {c['status'].value_counts().to_dict()}")
    print(f"Gaps: {int((gaps['n_missing'] > 0).sum())} campus(es) with missing fields -> outputs/gap_report.csv")

    if v.errors:
        print("\nNot writing GeoPackage until errors are fixed.")
        return 1

    gpkg = out_dir / "dct_projects.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    pts = c[(c["lat"] != "") & (c["lon"] != "")].copy()
    if len(pts):
        pts["lat"], pts["lon"] = pts["lat"].astype(float), pts["lon"].astype(float)
        g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts["lon"], pts["lat"]), crs="EPSG:4326")
        g.to_file(gpkg, layer="campuses", driver="GPKG")
    for name in ("phases", "milestones", "claims", "sources", "observations"):
        if len(t[name]):
            _attr_layer(t[name]).to_file(gpkg, layer=name, driver="GPKG")
    plan = imagery_plan(cfg, c)
    if len(plan):
        _attr_layer(plan).to_file(gpkg, layer="imagery_plan", driver="GPKG")
    fp = load_footprints(cfg)
    if fp is not None:
        fp.to_file(gpkg, layer="footprints", driver="GPKG")
    print(f"\nWrote {gpkg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
