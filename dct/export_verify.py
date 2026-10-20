"""Export the verification sheet for Google Earth: xlsx (campuses + phases), csv, kml.

Run: python -m dct.export_verify
You fill verified_lat / verified_lon / verified (Y/N/unsure) / verify_notes, then run
python -m dct.import_verify outputs/dct_sites_verify.xlsx
"""
from __future__ import annotations

import pandas as pd
import simplekml

from dct.common import REPO_ROOT, load_config, read_seed

VERIFY_COLUMNS = ["verified_lat", "verified_lon", "verified", "verify_notes"]


def _phase_summary(phases: pd.DataFrame, campus_id: str) -> str:
    sub = phases[phases["campus_id"] == campus_id]
    parts = []
    for r in sub.to_dict("records"):
        mw = f" {r['it_mw']} MW" if r["it_mw"] else ""
        parts.append(f"{r['name']}{mw} [{r['status']}]")
    return "; ".join(parts)


def _milestone_summary(phases: pd.DataFrame, miles: pd.DataFrame, campus_id: str) -> tuple[str, str]:
    """(announced start, target go-live) as the earliest groundbreaking and latest go_live dates."""
    pids = set(phases.loc[phases["campus_id"] == campus_id, "phase_id"])
    sub = miles[miles["phase_id"].isin(pids)]
    # latest non-superseded row per type
    superseded = set(sub["supersedes_id"]) - {""}
    sub = sub[~sub["milestone_id"].isin(superseded)]
    gb = sub[sub["type"] == "groundbreaking"]["date"].sort_values()
    gl = sub[sub["type"] == "go_live"]["date"].sort_values()
    return (gb.iloc[0] if len(gb) else "", gl.iloc[-1] if len(gl) else "")


def _links(lat: str, lon: str) -> tuple[str, str]:
    if not lat or not lon:
        return "", ""
    earth = f"https://earth.google.com/web/@{lat},{lon},0a,1500d,35y,0h,0t,0r"
    osm = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=17/{lat}/{lon}"
    return earth, osm


def build_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    campuses, phases = read_seed(cfg, "campuses"), read_seed(cfg, "phases")
    miles, sources = read_seed(cfg, "milestones"), read_seed(cfg, "sources")
    url_by_id = dict(zip(sources["source_id"], sources["url"]))

    rows = []
    for r in campuses.to_dict("records"):
        start, golive = _milestone_summary(phases, miles, r["campus_id"])
        earth, osm = _links(r["lat"], r["lon"])
        rows.append({
            "campus_id": r["campus_id"], "site_name": r["name"], "program": r["program"],
            "phases": _phase_summary(phases, r["campus_id"]),
            "owner_developer": r["owner"], "operator": r["operator"], "tenant": r["tenant"],
            "country": r["country"], "state": r["state"], "locality": r["locality"],
            "lat": r["lat"], "lon": r["lon"],
            "location_precision": r["location_precision"], "location_method": r["location_method"],
            "status": r["status"], "it_mw_total": r["it_mw_total"], "grid_mw": r["grid_mw"],
            "area_ha": r["area_ha"], "investment": r["investment_text"],
            "groundbreaking": start, "target_go_live": golive,
            "latest_news": r["latest_news"], "latest_news_date": r["latest_news_date"],
            "risk_note": r["risk_note"], "notes": r["notes"],
            "main_source_url": url_by_id.get(r["main_source_id"], ""),
            "osm_id": r["osm_id"], "google_earth_link": earth, "osm_link": osm,
            **{c: "" for c in VERIFY_COLUMNS},
        })
    camp_df = pd.DataFrame(rows)

    prow = []
    names = dict(zip(campuses["campus_id"], campuses["name"]))
    for r in phases.to_dict("records"):
        prow.append({"phase_id": r["phase_id"], "campus_id": r["campus_id"],
                     "site_name": names.get(r["campus_id"], ""), "phase": r["name"],
                     "it_mw": r["it_mw"], "status": r["status"], "notes": r["notes"]})
    return camp_df, pd.DataFrame(prow)


def main() -> None:
    cfg = load_config()
    out = REPO_ROOT / cfg["paths"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    camp, ph = build_tables(cfg)

    camp.to_csv(out / "dct_sites_verify.csv", index=False)
    xlsx = out / "dct_sites_verify.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        camp.to_excel(xw, sheet_name="campuses", index=False)
        ph.to_excel(xw, sheet_name="phases", index=False)
        for sheet, df in (("campuses", camp), ("phases", ph)):
            ws = xw.sheets[sheet]
            ws.freeze_panes = "C2"
            for i, col in enumerate(df.columns, start=1):
                width = min(60, max(12, int(df[col].astype(str).str.len().max() or 0) + 2, len(col) + 2))
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width

    kml = simplekml.Kml(name="DCT sites")
    n = 0
    for r in camp.to_dict("records"):
        if not r["lat"] or not r["lon"]:
            continue
        pnt = kml.newpoint(name=f"{r['site_name']} ({r['status']})", coords=[(float(r["lon"]), float(r["lat"]))])
        pnt.description = (
            f"Program: {r['program']}\nOwner: {r['owner_developer']} | Operator: {r['operator']} | Tenant: {r['tenant']}\n"
            f"Phases: {r['phases']}\nIT MW: {r['it_mw_total']} | Grid MW: {r['grid_mw']} | Area ha: {r['area_ha']}\n"
            f"Location: {r['location_precision']} via {r['location_method']}\n"
            f"Latest news ({r['latest_news_date']}): {r['latest_news']}\nRisk: {r['risk_note']}\n"
            f"Source: {r['main_source_url']}"
        )
        n += 1
    kml.save(str(out / "dct_sites_verify.kml"))
    print(f"Wrote {xlsx}\n      {out / 'dct_sites_verify.csv'}\n      {out / 'dct_sites_verify.kml'} ({n} placemarks)")


if __name__ == "__main__":
    main()
