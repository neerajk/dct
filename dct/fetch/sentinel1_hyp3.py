"""Sentinel-1 for every campus via ASF HyP3 (free tier: 8,000 credits/month, outputs kept 14 days).

Two products per month per site:
  RTC (gamma0, 20 m)   - backscatter, cue for steel/structure rising. One job covers a whole scene,
                         so one job serves every campus of a cluster.
  Burst InSAR (80 m)   - includes coherence, cue for clearing/earthworks. One burst pair per campus.

Subcommands (run from the repo root):
  python -m dct.fetch.sentinel1_hyp3 plan                 # search ASF, list jobs + credit estimate, submit nothing
  python -m dct.fetch.sentinel1_hyp3 submit --limit 50    # submit up to N planned jobs not yet submitted
  python -m dct.fetch.sentinel1_hyp3 download             # download finished jobs and clip to campus AOIs

Needs a free NASA Earthdata login in ~/.netrc. Re-run 'download' within 14 days of completion.
NOTE: written against asf_search / hyp3_sdk without being executed here; if a property name
differs in your installed version the error message will name it - fix it in _burst_fields().
"""
from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict

import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import Point, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from dct.common import (REPO_ROOT, campus_aoi_4326, campus_date_range, campuses_with_coords,
                        load_config, load_footprints, manifest_row, raw_dir)


# ---------- planning -------------------------------------------------------------------

def _jobs_path(cfg: dict):
    return REPO_ROOT / cfg["hyp3"]["jobs_file"]


def _load_jobs(cfg: dict) -> dict:
    p = _jobs_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {}


def _save_jobs(cfg: dict, jobs: dict) -> None:
    p = _jobs_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jobs, indent=2))


def _first_per_month(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """items = [(iso_date, name)] -> earliest per YYYY-MM."""
    seen, out = set(), []
    for d, name in sorted(items):
        if d[:7] not in seen:
            seen.add(d[:7])
            out.append((d, name))
    return out


def _rtc_jobs_for_cluster(cfg: dict, cluster: str, campuses: list[dict]) -> list[dict]:
    import asf_search as asf
    c0 = campuses[0]
    start = min(campus_date_range(cfg, c["campus_id"])[0] for c in campuses)
    end = campus_date_range(cfg, c0["campus_id"])[1]
    res = asf.geo_search(
        platform=[asf.PLATFORM.SENTINEL1], processingLevel=[asf.PRODUCT_TYPE.SLC],
        beamMode=asf.BEAMMODE.IW, intersectsWith=Point(c0["lon"], c0["lat"]).wkt,
        start=start, end=end, maxResults=3000,
    )
    # keep the single relative orbit + direction with the most acquisitions (consistent geometry)
    groups = Counter((r.properties["pathNumber"], r.properties["flightDirection"]) for r in res)
    if not groups:
        return []
    track = groups.most_common(1)[0][0]
    picked = [(r.properties["startTime"][:10], r.properties["sceneName"]) for r in res
              if (r.properties["pathNumber"], r.properties["flightDirection"]) == track]
    return [{"kind": "rtc", "cluster": cluster, "date": d, "granule": g,
             "name": f"dct_rtc_{cluster}_{d[:7]}", "credits": cfg["hyp3"]["credits"]["rtc_20m"]}
            for d, g in _first_per_month(picked)]


def _burst_fields(r) -> tuple[str, str, str]:
    """(full burst id, iso date, burst granule name) from an asf_search burst result."""
    p = r.properties
    burst = p.get("burst", {})
    full_id = burst.get("fullBurstID") or p.get("fullBurstID")
    name = p.get("fileID") or p.get("sceneName")
    if not (full_id and name):
        raise KeyError(f"burst properties missing fullBurstID/fileID; keys: {sorted(p)}")
    return full_id, p["startTime"][:10], name


def _insar_jobs_for_campus(cfg: dict, campus: dict) -> list[dict]:
    import asf_search as asf
    start, end = campus_date_range(cfg, campus["campus_id"])
    res = asf.geo_search(
        dataset=asf.DATASET.SLC_BURST, intersectsWith=Point(campus["lon"], campus["lat"]).wkt,
        polarization="VV", start=start, end=end, maxResults=3000,
    )
    by_id = defaultdict(list)
    for r in res:
        full_id, d, name = _burst_fields(r)
        by_id[full_id].append((d, name))
    if not by_id:
        return []
    best = max(by_id, key=lambda k: len(by_id[k]))          # burst id with most acquisitions
    acq = sorted(by_id[best])
    gap = cfg["hyp3"]["max_pair_gap_days"]
    pairs = []
    for (d1, g1), (d2, g2) in zip(acq, acq[1:]):
        if (pd.Timestamp(d2) - pd.Timestamp(d1)).days <= gap:
            pairs.append((d1, g1, g2))
    seen, out = set(), []
    for d1, g1, g2 in pairs:                                  # first pair of each month
        if d1[:7] in seen:
            continue
        seen.add(d1[:7])
        out.append({"kind": "insar", "campus_id": campus["campus_id"], "date": d1,
                    "granule": g1, "granule2": g2,
                    "name": f"dct_insar_{campus['campus_id']}_{d1[:7]}",
                    "credits": cfg["hyp3"]["credits"]["burst_insar_80m"]})
    return out


def plan(cfg: dict) -> list[dict]:
    campuses = campuses_with_coords(cfg)
    by_cluster = defaultdict(list)
    for c in campuses:
        by_cluster[c["cluster"]].append(c)
    jobs = []
    for cluster, cs in by_cluster.items():
        print(f"planning RTC for {cluster} ...")
        jobs += _rtc_jobs_for_cluster(cfg, cluster, cs)
    for c in campuses:
        print(f"planning burst InSAR for {c['campus_id']} ...")
        jobs += _insar_jobs_for_campus(cfg, c)
    total = sum(j["credits"] for j in jobs)
    print(f"\n{len(jobs)} job(s): "
          f"{sum(j['kind'] == 'rtc' for j in jobs)} RTC + {sum(j['kind'] == 'insar' for j in jobs)} InSAR; "
          f"estimated {total:.0f} credits (monthly free allowance 8,000)")
    state = _load_jobs(cfg)
    for j in jobs:
        state.setdefault(j["name"], {**j, "job_id": None})
    _save_jobs(cfg, state)
    return jobs


# ---------- submit / download -------------------------------------------------------------

def submit(cfg: dict, limit: int) -> None:
    import hyp3_sdk as sdk
    hyp3 = sdk.HyP3()
    state = _load_jobs(cfg)
    todo = [j for j in state.values() if not j["job_id"]][:limit]
    cost = sum(j["credits"] for j in todo)
    left = hyp3.check_credits_available()
    print(f"{len(todo)} job(s), {cost:.0f} credits; {left:.0f} available")
    if cost > left:
        raise SystemExit("not enough credits - lower --limit")
    h = cfg["hyp3"]
    for j in todo:
        if j["kind"] == "rtc":
            job = hyp3.submit_rtc_job(j["granule"], name=j["name"], resolution=h["rtc_resolution_m"])
        else:
            job = hyp3.submit_insar_isce_burst_job(j["granule"], j["granule2"], name=j["name"], looks=h["insar_looks"])
        j["job_id"] = job.job_id
        _save_jobs(cfg, state)
        print(f"  submitted {j['name']}")


def _clip(tif, aoi_4326, out_path) -> bool:
    with rasterio.open(tif) as src:
        fwd = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        geom = shp_transform(fwd, aoi_4326)
        try:
            data, transform = rio_mask(src, [mapping(geom)], crop=True)
        except ValueError:  # AOI does not overlap this raster
            return False
        profile = src.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2], transform=transform)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
    return True


def download(cfg: dict) -> list[dict]:
    import hyp3_sdk as sdk
    hyp3 = sdk.HyP3()
    state = _load_jobs(cfg)
    campuses = {c["campus_id"]: c for c in campuses_with_coords(cfg)}
    footprints = load_footprints(cfg)
    rows = []
    for name, j in state.items():
        if not j["job_id"] or j.get("downloaded"):
            continue
        job = hyp3.get_job_by_id(j["job_id"])
        if not job.succeeded():
            print(f"  {name}: {job.status_code}")
            continue
        zdir = raw_dir(cfg, "hyp3", "zips")
        files = job.download_files(zdir, create=True)
        targets = ([campuses[j["campus_id"]]] if j["kind"] == "insar" and j["campus_id"] in campuses
                   else [c for c in campuses.values() if c.get("cluster") == j.get("cluster")])
        want = "_corr.tif" if j["kind"] == "insar" else ("_VV.tif", "_VH.tif")
        with zipfile.ZipFile(files[0]) as z:
            members = [m for m in z.namelist() if m.endswith(want)]
            xdir = zdir / name
            z.extractall(xdir, members=members)
        for c in targets:
            aoi = campus_aoi_4326(cfg, c, footprints)
            for m in members:
                suffix = "coh" if j["kind"] == "insar" else m.rsplit("_", 1)[-1].replace(".tif", "").lower()
                out = raw_dir(cfg, "s1", c["campus_id"], f"{j['date'][:7]}_{j['kind']}_{suffix}.tif")
                ok = _clip(xdir / m, aoi, out)
                rows.append(manifest_row(c["campus_id"], f"sentinel1_{j['kind']}", j["date"], name,
                                         out if ok else None, "ok" if ok else "gap", "" if ok else "AOI outside raster"))
        j["downloaded"] = True
        _save_jobs(cfg, state)
        print(f"  downloaded + clipped {name}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["plan", "submit", "download"])
    ap.add_argument("--limit", type=int, default=50)
    a = ap.parse_args()
    cfg = load_config()
    if a.action == "plan":
        plan(cfg)
    elif a.action == "submit":
        submit(cfg, a.limit)
    else:
        rows = download(cfg)
        print(f"{len(rows)} clip(s) written")


if __name__ == "__main__":
    main()
