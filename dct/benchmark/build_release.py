"""Package DCT-US benchmark v0.

Run:  python -m dct.benchmark.build_release
Writes outputs/us/release_v0_<YYYY-MM-DD>/ :
  sites.geojson          campus outlines + attributes (OpenStreetMap contributors, ODbL)
  anchors.csv            Epoch / NAIP-auto / manual anchors, with a 'source' column
  monthly_labels.parquet (csv fallback) soft stage probabilities per campus-month (derived from the anchors)
  scenes.csv             Sentinel-2 item IDs and dates, OPERA granule names, NAIP item IDs: NO imagery is redistributed
  splits.json            leave-one-hub-out + temporal hold-out
  baseline_scores.csv    benchmark_scores.csv from dct.benchmark.evaluate, if present
  DATASHEET.md, LICENCES.md
Every table keeps its source and licence; decide the licence of the published bundle before release
(the outline layer is share-alike ODbL; Epoch rows are CC-BY and need credit to Epoch AI).
"""
from __future__ import annotations

import shutil
from datetime import date

import geopandas as gpd
import pandas as pd

from dct.benchmark.dataset import _concat_csv
from dct.benchmark.splits import build as build_splits
from dct.common import REPO_ROOT, load_us_config, out_dir, proc_dir, raw_dir, read_seed, require_footprints

LICENCES = """# Licences of the parts of DCT-US v0

| Part | Source | Licence / terms |
|---|---|---|
| sites.geojson | OpenStreetMap contributors via Overpass | ODbL 1.0 (attribution + share-alike for a published database) |
| anchors.csv, source=epoch | Epoch AI, Frontier Data Centers hub | CC BY 4.0 (credit Epoch AI) |
| anchors.csv, source=naip_auto / manual | derived from USDA NAIP imagery (public domain) and our own labelling | ours; imagery not included |
| monthly_labels | derived from the anchors | inherits the anchors' terms |
| scenes.csv | Copernicus Sentinel-2 (open), NASA OPERA RTC-S1 (CC BY 4.0), USDA NAIP (public domain) | identifiers only |
| splits.json, baseline_scores.csv | ours | to be decided |

Clay v1.5 weights are Apache-2.0 (Clay Foundation). Esri imagery is never used for training or distributed.
"""


def main() -> None:
    cfg = load_us_config()
    fp = require_footprints(cfg)
    dest = out_dir(cfg, f"release_v0_{date.today().isoformat()}")
    dest.mkdir(parents=True, exist_ok=True)

    # sites
    c = read_seed(cfg, "campuses")
    sites = fp.merge(c[["campus_id", "name", "operator", "state", "cluster", "osm_id", "data_source", "licence"]],
                     on="campus_id", how="left")
    sites.to_file(dest / "sites.geojson", driver="GeoJSON")

    # anchors
    parts = []
    for name in ("anchors_epoch.csv", "anchors_naip.csv"):
        p = proc_dir(cfg, name)
        if p.exists():
            parts.append(pd.read_csv(p).assign(file=name))
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(dest / "anchors.csv", index=False)

    # monthly labels
    lab = pd.read_csv(proc_dir(cfg, "monthly_labels.csv"))
    try:
        lab.to_parquet(dest / "monthly_labels.parquet", index=False)
    except Exception:
        lab.to_csv(dest / "monthly_labels.csv", index=False)

    # scene identifiers only
    scenes = []
    s2 = _concat_csv(proc_dir(cfg, "s2"))
    if len(s2):
        scenes.append(s2.loc[s2["status"] == "ok", ["campus_id", "month", "date", "item_id"]].assign(sensor="sentinel-2-l2a"))
    op = _concat_csv(proc_dir(cfg, "opera"))
    if len(op):
        scenes.append(op[["campus_id", "month", "date", "granule"]].rename(columns={"granule": "item_id"}).assign(sensor="opera-rtc-s1"))
    nm = raw_dir(cfg, "naip_manifest.csv")
    if nm.exists():
        n = pd.read_csv(nm, dtype=str)
        n = n[n["status"].isin(["ok", "cached"])]
        scenes.append(pd.DataFrame({"campus_id": n["campus_id"], "month": n["date"].str[:7], "date": n["date"],
                                    "item_id": n["item_id"], "sensor": "naip"}))
    if scenes:
        pd.concat(scenes, ignore_index=True).to_csv(dest / "scenes.csv", index=False)

    import json
    (dest / "splits.json").write_text(json.dumps(build_splits(cfg), indent=1))
    scores = out_dir(cfg, "benchmark_scores.csv")
    if scores.exists():
        shutil.copy(scores, dest / "baseline_scores.csv")
    shutil.copy(REPO_ROOT / "docs" / "DATASHEET.md", dest / "DATASHEET.md")
    (dest / "LICENCES.md").write_text(LICENCES)
    print(f"release written to {dest}")
    for p in sorted(dest.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
