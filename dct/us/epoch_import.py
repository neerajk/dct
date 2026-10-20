"""US benchmark step 2: Epoch AI timelines -> dated stage anchors (CC-BY, credit Epoch AI).

Run (repo root):  python -m dct.us.epoch_import
Needs data/seed_us (run dct.us.osm_sites first). Downloads data_centers.zip once to data/raw/us/epoch/.

Epoch publishes NO coordinates (data_centers.csv has Name, Country, Address, ...). Site positions are therefore
  1. taken from data/seed_us/epoch_site_coords.csv (site,lat,lon) if you provide it (wins), else
  2. geocoded from the Address with Nominatim (1 request/s, cached in data/raw/us/epoch/geocodes.csv).
Sites that fail to geocode are listed in outputs/us/epoch_geocode_failed.csv: add them to epoch_site_coords.csv
and re-run. Only United States sites are used (epoch.country_filter).

Outputs:
  data/processed/us/anchors_epoch.csv   anchor_id, campus_id, date, stage, uncertainty_months, source,
                                        reliability, epoch_site, status_text
  outputs/us/epoch_matches.csv          every Epoch site -> campus match (distance, names): REVIEW this
  outputs/us/epoch_unmapped.csv         status texts no keyword matched (map them by hand)

Stage rules (campus stage = the most advanced building):
  * text -> stage by keywords (whole words, highest stage wins) AFTER removing URLs/markdown links and negated
    phrases such as 'not fully operational';
  * 'Buildings operational' >= threshold forces stage 4; == 0 caps the stage at 3 (the number beats the prose).
Rows dated in the future are plans, not observations, and are skipped. An Epoch site is matched to the nearest
OSM campuses within site_match_radius_m (capped by max_campuses_per_site); a dense hub can mismatch.
"""
from __future__ import annotations

import io
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests
from pyproj import Geod

from dct.common import REPO_ROOT, load_us_config, out_dir, proc_dir, raw_dir, read_seed

GEOD = Geod(ellps="WGS84")
HTTP_TIMEOUT_S = 120
NOMINATIM = "https://nominatim.openstreetmap.org/search"
URL_OR_LINK = re.compile(r"https?://\S+|\[[^\]]*\]\([^)]*\)")
NEGATED = re.compile(
    r"\b(?:not|non|never|isn't|aren't|wasn't|yet to|unlikely|no)\b[^.;:\n]{0,50}?"
    r"\b(?:fully\s+)?(?:operational|operating|online|energi[sz]ed|in service|complete[d]?)\b")


def download_epoch(cfg: dict) -> dict[str, pd.DataFrame]:
    """Return {csv name: DataFrame} for data_centers.csv and data_center_timelines.csv."""
    dest = raw_dir(cfg, "epoch")
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "data_centers.zip"
    if not zpath.exists():
        r = requests.get(cfg["us"]["epoch"]["data_url"], timeout=HTTP_TIMEOUT_S)
        r.raise_for_status()
        zpath.write_bytes(r.content)
    out = {}
    with zipfile.ZipFile(zpath) as zf:
        names = {n.rsplit("/", 1)[-1]: n for n in zf.namelist()}
        for want in ("data_centers.csv", "data_center_timelines.csv"):
            if want not in names:
                raise SystemExit(f"{want} not found in {zpath}; members: {sorted(names)}")
            with zf.open(names[want]) as fh:               # read in memory, never extract paths
                out[want] = pd.read_csv(io.BytesIO(fh.read()))
    return out


def _find_col(df: pd.DataFrame, configured: str | None, *needles: str, what: str) -> str:
    """Configured name, else an exact (case-insensitive) match, else a substring match of needles >= 4 chars."""
    if configured:
        if configured not in df.columns:
            raise SystemExit(f"epoch.columns.{what}='{configured}' not in columns: {list(df.columns)}")
        return configured
    low = {str(c).lower(): c for c in df.columns}
    for n in needles:
        if n in low:
            return low[n]
    for n in needles:
        if len(n) >= 4:
            for lc, c in low.items():
                if n in lc:
                    return c
    raise SystemExit(f"Could not auto-detect the Epoch '{what}' column. Columns: {list(df.columns)}. "
                     f"Set epoch.columns.{what} in config/us_benchmark.yaml.")


def stage_from_text(text: str, keywords: dict[int, list[str]]) -> int | None:
    t = NEGATED.sub(" ", URL_OR_LINK.sub(" ", str(text).lower()))
    for stage in sorted(keywords, reverse=True):
        if any(re.search(rf"\b{re.escape(k)}\b", t) for k in keywords[stage]):
            return int(stage)
    return None


def site_positions(cfg: dict, sites_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sites with lat/lon, failures). Overrides win, then the cache, then Nominatim. A site with no address cannot be
    geocoded (never send 'nan' to the geocoder): it is listed as a failure for epoch_site_coords.csv."""
    ecfg = cfg["us"]["epoch"]
    cols = ecfg["columns"]
    name_c = _find_col(sites_raw, cols["site_name"], "name", "title", what="site_name")
    addr_c = _find_col(sites_raw, cols["address"], "address", what="address")
    df = sites_raw
    if ecfg.get("country_filter"):
        ctry_c = _find_col(sites_raw, cols["country"], "country", what="country")
        df = sites_raw[sites_raw[ctry_c].astype(str).str.strip() == ecfg["country_filter"]]
    sites = pd.DataFrame({"site": df[name_c].astype(str),
                          "address": df[addr_c].fillna("").astype(str).str.strip()}).drop_duplicates("site")

    override = {}
    op = REPO_ROOT / ecfg["coords_override"]
    if op.exists():
        o = pd.read_csv(op)
        override = {r.site: (float(r.lat), float(r.lon)) for r in o.itertuples()}
    cache_p = REPO_ROOT / ecfg["geocode_cache"]
    cache = {}
    if cache_p.exists():
        c = pd.read_csv(cache_p)
        cache = {r.site: r for r in c.itertuples()}

    rows, fails, new_cache = [], [], []
    headers = {"User-Agent": ecfg["geocode_user_agent"]}
    try:
        for r in sites.itertuples():
            lat = lon = np.nan
            if r.site in override:
                lat, lon = override[r.site]
            elif not r.address:
                pass                                         # no address and no override: reported as a failure below
            elif r.site in cache and cache[r.site].status == "ok":
                lat, lon = float(cache[r.site].lat), float(cache[r.site].lon)
            else:
                try:
                    resp = requests.get(NOMINATIM, params={"q": r.address, "format": "json", "limit": 1, "countrycodes": "us"},
                                        headers=headers, timeout=30)
                    if resp.status_code in (403, 429):
                        raise SystemExit(f"Nominatim refused the request (HTTP {resp.status_code}): wait, set a more specific "
                                         f"epoch.geocode_user_agent, or supply {ecfg['coords_override']}. Results so far are cached.")
                    resp.raise_for_status()
                    hit = resp.json()
                    if isinstance(hit, list) and hit:
                        lat, lon = float(hit[0]["lat"]), float(hit[0]["lon"])
                except (requests.RequestException, ValueError, KeyError, IndexError):
                    pass
                new_cache.append({"site": r.site, "address": r.address, "lat": lat, "lon": lon,
                                  "status": "ok" if not np.isnan(lat) else "failed"})
                time.sleep(1.1)                              # Nominatim usage policy: at most 1 request per second
            if np.isnan(lat):
                fails.append({"site": r.site, "address": r.address})
            else:
                rows.append({"site": r.site, "lat": lat, "lon": lon})
    finally:                                                 # keep what was geocoded even on Ctrl-C or a refusal
        if new_cache:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            old = pd.read_csv(cache_p) if cache_p.exists() else pd.DataFrame(columns=["site", "address", "lat", "lon", "status"])
            pd.concat([old, pd.DataFrame(new_cache)]).drop_duplicates("site", keep="last").to_csv(cache_p, index=False)
    return pd.DataFrame(rows, columns=["site", "lat", "lon"]), pd.DataFrame(fails, columns=["site", "address"])


def match_sites(sites: pd.DataFrame, campuses: pd.DataFrame, radius_m: float, cap: int) -> pd.DataFrame:
    clat = campuses["lat"].astype(float).to_numpy()
    clon = campuses["lon"].astype(float).to_numpy()
    rows = []
    for _, s in sites.iterrows():
        _, _, dist = GEOD.inv(np.full(len(clat), s["lon"]), np.full(len(clat), s["lat"]), clon, clat)
        order = np.argsort(dist)[:cap]
        for idx in order:
            if dist[idx] <= radius_m:
                c = campuses.iloc[idx]
                rows.append({"epoch_site": s["site"], "campus_id": c["campus_id"], "campus_name": c["name"],
                             "distance_m": round(float(dist[idx]), 0)})
    return pd.DataFrame(rows, columns=["epoch_site", "campus_id", "campus_name", "distance_m"])


def main() -> None:
    cfg = load_us_config()
    ecfg = cfg["us"]["epoch"]
    cols = ecfg["columns"]
    data = download_epoch(cfg)
    sites_raw, tl = data["data_centers.csv"], data["data_center_timelines.csv"]
    o = out_dir(cfg)
    o.mkdir(parents=True, exist_ok=True)

    campuses = read_seed(cfg, "campuses")
    if campuses.empty:
        raise SystemExit("data/seed_us/campuses.csv is empty: run python -m dct.us.osm_sites first")
    sites, fails = site_positions(cfg, sites_raw)
    fails.to_csv(o / "epoch_geocode_failed.csv", index=False)
    if sites.empty:
        raise SystemExit("no Epoch site could be positioned; see outputs/us/epoch_geocode_failed.csv and supply "
                         f"{ecfg['coords_override']} (columns site,lat,lon)")
    print(f"{len(sites)} Epoch sites positioned, {len(fails)} failed (outputs/us/epoch_geocode_failed.csv)")
    matches = match_sites(sites, campuses, ecfg["site_match_radius_m"], ecfg["max_campuses_per_site"])
    matches.to_csv(o / "epoch_matches.csv", index=False)
    print(f"{matches['epoch_site'].nunique()} of {len(sites)} Epoch sites matched to {matches['campus_id'].nunique()} campuses")

    tsite = _find_col(tl, cols["timeline_site"], "data center", "site", what="timeline_site")
    tdate = _find_col(tl, cols["date"], "date", what="date")
    tstatus = _find_col(tl, cols["status"], "construction status", "status", what="status")
    tbuild = cols["buildings_operational"] if cols["buildings_operational"] in tl.columns else None
    keywords = {int(k): v for k, v in ecfg["keywords"].items()}
    orphan = sorted(set(tl[tsite].astype(str)) - set(sites_raw[_find_col(sites_raw, cols["site_name"], "name", "title", what="site_name")].astype(str)))
    if orphan:
        print(f"WARNING: {len(orphan)} timeline site name(s) not found in data_centers.csv, e.g. {orphan[:3]}. Their rows are skipped.")

    today = pd.Timestamp.today().normalize()
    thr = ecfg["operational_if_buildings_operational_ge"]
    by_site = matches.groupby("epoch_site")["campus_id"].apply(list).to_dict()
    anchors, unmapped, n_future = [], [], 0
    for _, r in tl.iterrows():
        site = str(r[tsite])
        if site not in by_site:
            continue
        d = pd.to_datetime(r[tdate], errors="coerce")
        if pd.isna(d):
            continue
        if d > today:
            n_future += 1
            continue
        stage = stage_from_text(r[tstatus], keywords)
        if tbuild is not None:
            b = pd.to_numeric(r[tbuild], errors="coerce")
            if pd.notna(b):
                if b >= thr:
                    stage = 4
                elif b == 0 and stage is not None:
                    stage = min(stage, 3)
        if stage is None:
            unmapped.append({"epoch_site": site, "date": d.date().isoformat(), "status_text": r[tstatus]})
            continue
        for cid in by_site[site]:
            anchors.append({
                "campus_id": cid, "date": d.date().isoformat(), "stage": stage,
                "uncertainty_months": ecfg["date_uncertainty_months"], "source": "epoch",
                "reliability": cfg["us"]["labels"]["source_reliability"]["epoch"],
                "epoch_site": site, "status_text": r[tstatus],
            })
    df = pd.DataFrame(anchors, columns=["campus_id", "date", "stage", "uncertainty_months", "source",
                                        "reliability", "epoch_site", "status_text"])
    df.insert(0, "anchor_id", [f"EPOCH_{i:05d}" for i in range(1, len(df) + 1)])
    p = proc_dir(cfg)
    p.mkdir(parents=True, exist_ok=True)
    df.to_csv(p / "anchors_epoch.csv", index=False)
    pd.DataFrame(unmapped, columns=["epoch_site", "date", "status_text"]).to_csv(o / "epoch_unmapped.csv", index=False)
    print(f"{len(df)} Epoch anchors on {df['campus_id'].nunique()} campuses -> {p / 'anchors_epoch.csv'}")
    print(f"skipped: {n_future} future-dated rows (plans), {len(unmapped)} unmapped texts -> epoch_unmapped.csv")
    print("Review outputs/us/epoch_matches.csv before trusting the anchors.")


if __name__ == "__main__":
    main()
