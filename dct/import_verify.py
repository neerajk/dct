"""Apply your Google Earth verification back into data/seed/campuses.csv.

Run: python -m dct.import_verify outputs/dct_sites_verify.xlsx

Per row of the 'campuses' sheet:
  verified = Y/yes  and verified_lat/lon filled -> lat/lon replaced, precision 'exact',
                       method 'user_verified', old position logged as a claim (field lat_lon_previous)
  verified = Y/yes  and verified_lat/lon blank   -> current lat/lon accepted: precision 'exact', method 'user_verified'
  verified = N      -> position kept, a note is appended to campuses.notes
  anything else     -> untouched (verify_notes still appended if present)
A backup campuses.csv.bak is written first.
"""
from __future__ import annotations

import shutil
import sys
from datetime import date

import pandas as pd

from dct.common import load_config, read_seed, seed_path, write_seed

USER_SOURCE_ID = "SRC_USER_VERIFY"


def main(xlsx_path: str) -> None:
    cfg = load_config()
    sheet = pd.read_excel(xlsx_path, sheet_name="campuses", dtype=str).fillna("")
    campuses = read_seed(cfg, "campuses")
    claims = read_seed(cfg, "claims")
    sources = read_seed(cfg, "sources")
    shutil.copy(seed_path(cfg, "campuses"), str(seed_path(cfg, "campuses")) + ".bak")

    if USER_SOURCE_ID not in set(sources["source_id"]):
        sources.loc[len(sources)] = {
            "source_id": USER_SOURCE_ID, "url": "manual://google-earth-verification",
            "publisher": "Analyst (Neeraj)", "title": "Manual verification in Google Earth",
            "pub_date": "", "accessed": date.today().isoformat(), "tier": "1",
        }

    idx = campuses.set_index("campus_id")
    n_upd = n_acc = n_rej = 0
    new_claims = []
    for r in sheet.to_dict("records"):
        cid = r["campus_id"]
        if cid not in idx.index:
            print(f"skip unknown campus_id {cid}")
            continue
        v = r["verified"].strip().lower()
        vlat, vlon = r["verified_lat"].strip(), r["verified_lon"].strip()
        note = r["verify_notes"].strip()
        if v in ("y", "yes"):
            if vlat and vlon:
                float(vlat), float(vlon)  # raises if not numeric
                old = f"{idx.at[cid, 'lat']},{idx.at[cid, 'lon']}"
                idx.at[cid, "lat"], idx.at[cid, "lon"] = vlat, vlon
                new_claims.append({
                    "claim_id": f"CL_VERIFY_{cid}", "entity_type": "campus", "entity_id": cid,
                    "field": "lat_lon_previous", "value": old, "unit": "deg",
                    "as_of": date.today().isoformat(), "source_id": USER_SOURCE_ID,
                })
                n_upd += 1
            else:
                n_acc += 1
            idx.at[cid, "location_precision"] = "exact"
            idx.at[cid, "location_method"] = "user_verified"
        elif v in ("n", "no"):
            n_rej += 1
            note = ("NOT VERIFIED: " + note).strip()
        if note:
            prev = idx.at[cid, "notes"]
            idx.at[cid, "notes"] = (prev + " | " if prev else "") + f"[{date.today().isoformat()}] {note}"

    write_seed(cfg, "campuses", idx.reset_index()[list(campuses.columns)])
    if new_claims:
        existing = set(claims["claim_id"])
        add = pd.DataFrame([c for c in new_claims if c["claim_id"] not in existing], columns=claims.columns)
        claims = pd.concat([claims, add], ignore_index=True)
    write_seed(cfg, "claims", claims)
    write_seed(cfg, "sources", sources)
    print(f"Updated positions: {n_upd} | accepted as-is: {n_acc} | marked not verified: {n_rej}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m dct.import_verify <verification.xlsx>")
    main(sys.argv[1])
