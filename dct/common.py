"""Shared helpers: config, seed-table IO, campus AOIs, date ranges, manifest rows.

Pattern follows potapulse/common.py (REPO_ROOT + load_config), plus the campus
helpers every fetcher needs.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import yaml
from pyproj import Transformer
from shapely.geometry import Point, box
from shapely.ops import transform as shp_transform

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "dct.yaml"

SEED_TABLES = ["campuses", "phases", "milestones", "claims", "sources", "observations"]

MANIFEST_COLUMNS = ["campus_id", "sensor", "date", "item_id", "path", "status", "note"]


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


US_CONFIG = REPO_ROOT / "config" / "us_benchmark.yaml"


def load_us_config() -> dict[str, Any]:
    """Base config (dct.yaml) merged with the US benchmark settings (us_benchmark.yaml).

    - cfg['paths'] is replaced by the US paths, so read_seed / raw_dir / build_db work on data/seed_us etc.
    - the five hubs are added to cfg['clusters'] (country US) so campus validation accepts them.
    - the whole US file is available as cfg['us'].
    """
    cfg = load_config()
    us = load_config(US_CONFIG)
    for key, hub in us["hubs"].items():
        cfg["clusters"][key] = {"country": "US", "bbox": hub["bbox"]}
    cfg["paths"] = us["paths"]
    cfg["us"] = us
    return cfg


def proc_dir(cfg: dict, *parts: str) -> Path:
    """data/processed/us/... (anchors, labels, embeddings). Only valid for the US config."""
    p = REPO_ROOT / cfg["us"]["paths"]["proc_dir"]
    for part in parts:
        p = p / part
    return p


def out_dir(cfg: dict, *parts: str) -> Path:
    p = REPO_ROOT / cfg["paths"]["out_dir"]
    for part in parts:
        p = p / part
    return p


def seed_path(cfg: dict, name: str) -> Path:
    return REPO_ROOT / cfg["paths"]["seed_dir"] / f"{name}.csv"


def read_seed(cfg: dict, name: str) -> pd.DataFrame:
    """Read a seed CSV with every column as a string ('' for blanks)."""
    return pd.read_csv(seed_path(cfg, name), dtype=str, keep_default_na=False)


def write_seed(cfg: dict, name: str, df: pd.DataFrame) -> None:
    df.to_csv(seed_path(cfg, name), index=False)


def load_footprints(cfg: dict) -> gpd.GeoDataFrame | None:
    """Traced campus outlines (EPSG:4326, column campus_id), or None if the file is absent."""
    path = REPO_ROOT / cfg["paths"]["seed_dir"] / "footprints.gpkg"
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    return gdf.to_crs("EPSG:4326") if len(gdf) else None


def require_footprints(cfg: dict) -> gpd.GeoDataFrame:
    """load_footprints, but stop with a clear message instead of returning None."""
    fp = load_footprints(cfg)
    if fp is None:
        raise SystemExit(f"no footprints in {cfg['paths']['seed_dir']} (footprints.gpkg missing or empty): "
                         "run python -m dct.us.osm_sites first")
    return fp


def utm_epsg(lon: float, lat: float) -> int:
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def _to_float(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def campuses_with_coords(cfg: dict) -> list[dict]:
    """Campus rows (as dicts) that have usable lat/lon."""
    df = read_seed(cfg, "campuses")
    rows = []
    for rec in df.to_dict("records"):
        lat, lon = _to_float(rec.get("lat")), _to_float(rec.get("lon"))
        if lat is None or lon is None:
            continue
        rec["lat"], rec["lon"] = lat, lon
        rows.append(rec)
    return rows


def campus_aoi_4326(cfg: dict, campus: dict, footprints: gpd.GeoDataFrame | None, kind: str = "point"):
    """AOI polygon (EPSG:4326) for one campus.

    Footprint (if traced) + footprint_buffer_m, otherwise the point + point_buffer_m.
    kind='aerial' uses aerial_buffer_m for the point case (many tiles, so smaller).
    """
    acfg = cfg["aoi"]
    lon, lat = campus["lon"], campus["lat"]
    utm = utm_epsg(lon, lat)
    fwd = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform
    inv = Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform

    geom = None
    if footprints is not None:
        match = footprints[footprints["campus_id"] == campus["campus_id"]]
        if len(match):
            geom = match.geometry.union_all()
    if geom is not None:
        buf = acfg["footprint_buffer_m"]
        g_utm = shp_transform(fwd, geom)
    else:
        buf = acfg["aerial_buffer_m"] if kind == "aerial" else acfg["point_buffer_m"]
        g_utm = shp_transform(fwd, Point(lon, lat))
    minx, miny, maxx, maxy = g_utm.buffer(buf).bounds
    return shp_transform(inv, box(minx, miny, maxx, maxy))


def campus_date_range(cfg: dict, campus_id: str) -> tuple[str, str]:
    """(start, end) ISO dates: earliest milestone - N months, floored at start_floor, to today."""
    floor = pd.Timestamp(cfg["time"]["start_floor"])
    phases = read_seed(cfg, "phases")
    miles = read_seed(cfg, "milestones")
    pids = set(phases.loc[phases["campus_id"] == campus_id, "phase_id"])
    dates = pd.to_datetime(miles.loc[miles["phase_id"].isin(pids), "date"], errors="coerce").dropna()
    start = floor
    if len(dates):
        start = max(floor, dates.min() - pd.DateOffset(months=cfg["time"]["months_before_first_milestone"]))
    start = start.replace(day=1)
    return start.strftime("%Y-%m-%d"), date.today().strftime("%Y-%m-%d")


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start).replace(day=1), pd.Timestamp(end), freq="MS"))


def raw_dir(cfg: dict, *parts: str) -> Path:
    p = REPO_ROOT / cfg["paths"]["raw_dir"]
    for part in parts:
        p = p / part
    return p


def manifest_row(campus_id, sensor, date_, item_id, path, status="ok", note="") -> dict:
    return {
        "campus_id": campus_id, "sensor": sensor, "date": date_, "item_id": item_id,
        "path": str(path) if path else "", "status": status, "note": note,
    }
