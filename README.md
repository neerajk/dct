# DCT - DataCenter Tracking

Satellite-based tracking of data-center construction (stage, delay, risk) for lenders and
investors in India and the EU. (DCT is the product; Drishti is the separate consultancy.)

Step 0 (this repo state): master project database for the first 20 sites + data fetch.

## Setup
```bash
micromamba create -f environment.yml
micromamba activate dct
```

## Workflow (run from the repo root)
```bash
# 1. validate seed tables, write outputs/dct_projects.gpkg + outputs/gap_report.csv
python -m dct.build_db

# 2. export the verification sheet (xlsx + csv + kml for Google Earth)
python -m dct.export_verify

# 3. after you fill verified_lat / verified_lon / verified in the xlsx
python -m dct.import_verify outputs/dct_sites_verify.xlsx

# 4. OSM candidate outlines (EU mostly)
python -m dct.fetch.osm_footprints

# 5. imagery (skip-if-cached, safe to re-run). LISS-IV is not included yet.
python -m dct.fetch.run_all                 # Sentinel-2, Landsat LST, NL + Hessen aerial
python -m dct.fetch.sentinel1_hyp3 plan     # Sentinel-1 job plan + credit estimate (no submit)
python -m dct.fetch.sentinel1_hyp3 submit --limit 50
python -m dct.fetch.sentinel1_hyp3 download # within 14 days of job completion!
```

Auth: HyP3 needs a free NASA Earthdata login stored in `~/.netrc`
(`machine urs.earthdata.nasa.gov login <user> password <pass>`).

## Data rules
- `data/seed/*.csv` is the master copy (hand-edited, git-tracked).
- Esri World Imagery is view-only (see `docs/view_layers.md`), never downloaded or used for training.
- Every MW, investment, area figure needs a row in `claims.csv` pointing to `sources.csv`.

---

## US benchmark v0 (Clay v1.5 via TerraTorch)

Docs: `docs/US_benchmark_methodology.md` (data flow and method), `docs/DCT_references.docx` (95 references),
`docs/DATASHEET.md`. Config: `config/us_benchmark.yaml`. **None of this code has been run yet** (written and reviewed by reading only).

### Environment (ML)
```bash
micromamba create -f environment-ml.yml && micromamba activate dct-ml   # terratorch >= 1.2.12
export KMP_DUPLICATE_LIB_OK=TRUE                                         # OpenMP clash on this Mac
```
Clay weights: `models/clay/clay-v1.5.ckpt` (moved from clay_LULC; a symlink was left there). Clay sensor table: `config/clay/metadata.yaml`.

### Run order (repo root)
```bash
# 1 sites (OSM, 5 hubs) -> data/seed_us, then validate
python -m dct.us.osm_sites
python -m dct.build_db --us

# 2 Epoch AI anchors (CC-BY); review outputs/us/epoch_matches.csv and epoch_unmapped.csv
python -m dct.us.epoch_import

# 3 NAIP chips (2010-2023 from Planetary Computer). For 2024-26 download USDA county mosaics by hand into
#   data/raw/us/naip_usda/ (see docstring of dct/us/naip_chips.py), then re-run.
python -m dct.us.naip_chips

# 4 NAIP teacher: embed, label ~200 campus-years by hand, train, repeat (active learning)
python -m dct.clay.naip_embed
python -m dct.clay.autolabel_naip                                  # first run writes label_queue.csv
PYTHONPATH=. streamlit run dct/labels/labeler_app.py               # label, save
python -m dct.clay.autolabel_naip                                  # trains, writes anchors_naip.csv + next queue

# 5 monthly labels (stages only go forward)
python -m dct.labels.build_monthly

# 6 Sentinel-2 features + Clay embeddings (use --no-clay for features only), OPERA radar
python -m dct.us.s2_extract
python -m dct.us.opera_rtc                                          # needs ~/.netrc Earthdata login

# 7 models A / B / C, leave-one-hub-out + temporal hold-out, and the release bundle
python -m dct.benchmark.evaluate
python -m dct.benchmark.build_release
```
Smoke tests first: add `--hub us_nova --limit 5` to steps 3, 4 (naip_embed), 6.
