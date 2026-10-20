"""Table schemas + cross-table validation for the seed CSVs.

Row models coerce types (blank string -> None). Enumerations come from
config/dct.yaml, not from Literal types, so adding a status needs no code change.
Rule of the database: every MW / investment / area figure on a campus or phase needs a
claim; every milestone and claim needs a source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from pydantic import BaseModel, field_validator

from dct.common import SEED_TABLES, load_footprints, read_seed


class _Row(BaseModel):
    @field_validator("*", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        return None if isinstance(v, str) and v.strip() == "" else v


class Campus(_Row):
    campus_id: str
    name: str
    program: Optional[str] = None
    owner: Optional[str] = None
    operator: Optional[str] = None
    tenant: Optional[str] = None
    country: str
    state: Optional[str] = None
    locality: Optional[str] = None
    cluster: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    location_precision: Optional[str] = None
    location_method: Optional[str] = None
    area_ha: Optional[float] = None
    status: str
    it_mw_total: Optional[float] = None
    grid_mw: Optional[float] = None
    investment_text: Optional[str] = None
    osm_id: Optional[str] = None
    latest_news: Optional[str] = None
    latest_news_date: Optional[str] = None
    risk_note: Optional[str] = None
    main_source_id: Optional[str] = None
    notes: Optional[str] = None
    data_source: Optional[str] = None   # manual | osm  (licence is decided per source)
    licence: Optional[str] = None       # n/a | ODbL


class Phase(_Row):
    phase_id: str
    campus_id: str
    name: str
    it_mw: Optional[float] = None
    status: str
    notes: Optional[str] = None


class Milestone(_Row):
    milestone_id: str
    phase_id: str
    type: str
    date: str
    date_precision: str
    is_revised: Optional[str] = None
    supersedes_id: Optional[str] = None
    source_id: str
    note: Optional[str] = None


class Claim(_Row):
    claim_id: str
    entity_type: str
    entity_id: str
    field: str
    value: str
    unit: Optional[str] = None
    as_of: Optional[str] = None
    source_id: str


class Source(_Row):
    source_id: str
    url: str
    publisher: Optional[str] = None
    title: Optional[str] = None
    pub_date: Optional[str] = None
    accessed: Optional[str] = None
    tier: str


class Observation(_Row):
    obs_id: str
    phase_id: str
    date: str
    sensor: str
    scene_id: Optional[str] = None
    stage: Optional[str] = None
    confidence: Optional[float] = None
    method: Optional[str] = None
    note: Optional[str] = None


MODELS = {
    "campuses": Campus, "phases": Phase, "milestones": Milestone,
    "claims": Claim, "sources": Source, "observations": Observation,
}

# campus/phase fields that must be backed by at least one claim when filled
CLAIM_REQUIRED = {
    "campus": ["it_mw_total", "grid_mw", "investment_text", "area_ha"],
    "phase": ["it_mw"],
}


@dataclass
class Validation:
    tables: dict[str, pd.DataFrame]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _in_bbox(lon: float, lat: float, bbox: list[float]) -> bool:
    w, s, e, n = bbox
    return w <= lon <= e and s <= lat <= n


def validate(cfg: dict) -> Validation:
    tables = {name: read_seed(cfg, name) for name in SEED_TABLES}
    v = Validation(tables=tables)
    err, warn = v.errors.append, v.warnings.append

    # 1. row-level type validation + unique ids
    id_col = {"campuses": "campus_id", "phases": "phase_id", "milestones": "milestone_id",
              "claims": "claim_id", "sources": "source_id", "observations": "obs_id"}
    for name, df in tables.items():
        model = MODELS[name]
        missing = [c for c in model.model_fields if c not in df.columns]
        if missing:
            err(f"{name}.csv: missing columns {missing}")
            continue
        for i, rec in enumerate(df.to_dict("records")):
            try:
                model(**{k: rec[k] for k in model.model_fields})
            except Exception as exc:  # pydantic ValidationError
                err(f"{name}.csv row {i + 2}: {exc}")
        dup = df[id_col[name]][df[id_col[name]].duplicated()].tolist()
        if dup:
            err(f"{name}.csv: duplicate {id_col[name]} {sorted(set(dup))}")

    campuses, phases, miles = tables["campuses"], tables["phases"], tables["milestones"]
    claims, sources = tables["claims"], tables["sources"]
    en = cfg["enums"]

    def check_enum(df, col, allowed, tbl, blank_ok=True):
        for i, val in enumerate(df[col]):
            if val == "" and blank_ok:
                continue
            if val not in allowed:
                err(f"{tbl}.csv row {i + 2}: {col}='{val}' not in {allowed}")

    # 2. enums
    check_enum(campuses, "country", en["country"], "campuses", blank_ok=False)
    check_enum(campuses, "status", en["status"], "campuses", blank_ok=False)
    check_enum(campuses, "location_precision", en["location_precision"], "campuses")
    check_enum(campuses, "location_method", en["location_method"], "campuses")
    check_enum(campuses, "cluster", list(cfg["clusters"]), "campuses", blank_ok=False)
    check_enum(phases, "status", en["status"], "phases", blank_ok=False)
    check_enum(miles, "type", en["milestone_type"], "milestones", blank_ok=False)
    check_enum(miles, "date_precision", en["date_precision"], "milestones", blank_ok=False)
    check_enum(sources, "tier", en["source_tier"], "sources", blank_ok=False)
    check_enum(claims, "entity_type", en["entity_type"], "claims", blank_ok=False)

    # 3. foreign keys
    campus_ids, phase_ids = set(campuses["campus_id"]), set(phases["phase_id"])
    source_ids, milestone_ids = set(sources["source_id"]), set(miles["milestone_id"])
    for i, r in enumerate(phases.to_dict("records")):
        if r["campus_id"] not in campus_ids:
            err(f"phases.csv row {i + 2}: unknown campus_id '{r['campus_id']}'")
    for i, r in enumerate(miles.to_dict("records")):
        if r["phase_id"] not in phase_ids:
            err(f"milestones.csv row {i + 2}: unknown phase_id '{r['phase_id']}'")
        if r["source_id"] not in source_ids:
            err(f"milestones.csv row {i + 2}: unknown source_id '{r['source_id']}'")
        if r["is_revised"].lower() == "true" and r["supersedes_id"] not in milestone_ids:
            err(f"milestones.csv row {i + 2}: is_revised=true needs a valid supersedes_id")
    entity_sets = {"campus": campus_ids, "phase": phase_ids, "milestone": milestone_ids}
    for i, r in enumerate(claims.to_dict("records")):
        if r["source_id"] not in source_ids:
            err(f"claims.csv row {i + 2}: unknown source_id '{r['source_id']}'")
        if r["entity_type"] in entity_sets and r["entity_id"] not in entity_sets[r["entity_type"]]:
            err(f"claims.csv row {i + 2}: unknown {r['entity_type']} '{r['entity_id']}'")
    for i, r in enumerate(campuses.to_dict("records")):
        if r["main_source_id"] and r["main_source_id"] not in source_ids:
            err(f"campuses.csv row {i + 2}: unknown main_source_id '{r['main_source_id']}'")

    # 4. coordinates
    footprints = load_footprints(cfg)
    fp_ids = set(footprints["campus_id"]) if footprints is not None else set()
    for i, r in enumerate(campuses.to_dict("records")):
        row = i + 2
        lat_s, lon_s = r["lat"], r["lon"]
        if (lat_s == "") != (lon_s == ""):
            err(f"campuses.csv row {row}: lat and lon must both be set or both blank")
            continue
        if lat_s == "":
            continue
        try:
            lat, lon = float(lat_s), float(lon_s)
        except ValueError:
            continue  # already reported by the row model
        if not r["location_precision"]:
            err(f"campuses.csv row {row}: lat/lon set but location_precision blank")
        if r["cluster"] in cfg["clusters"] and not _in_bbox(lon, lat, cfg["clusters"][r["cluster"]]["bbox"]):
            warn(f"campuses.csv row {row}: ({lat}, {lon}) outside cluster '{r['cluster']}' bbox (swapped lat/lon?)")
        if r["location_precision"] == "exact" and r["campus_id"] not in fp_ids and r["location_method"] != "user_verified":
            err(f"campuses.csv row {row}: 'exact' needs a footprint or location_method=user_verified")

    # 5. claims back every figure
    have = {(c["entity_type"], c["entity_id"], c["field"]) for c in claims.to_dict("records")}
    for r in campuses.to_dict("records"):
        for f in CLAIM_REQUIRED["campus"]:
            if r[f] and ("campus", r["campus_id"], f) not in have:
                err(f"campus {r['campus_id']}: {f}='{r[f]}' has no claim")
    for r in phases.to_dict("records"):
        for f in CLAIM_REQUIRED["phase"]:
            if r[f] and ("phase", r["phase_id"], f) not in have:
                err(f"phase {r['phase_id']}: {f}='{r[f]}' has no claim")

    # 6. soft warnings
    for r in campuses.to_dict("records"):
        if not any(p["campus_id"] == r["campus_id"] for p in phases.to_dict("records")):
            warn(f"campus {r['campus_id']}: no phases yet")
    return v
