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

### Windows + NVIDIA GPU (current target: RTX 5050)
The code runs on whichever device is available. Order of choice: `--device` flag, `DCT_DEVICE` env var, `clay.device` in
`config/us_benchmark.yaml`, then auto (cuda > mps > cpu). A device you ask for but do not have is an error, never a silent CPU run.
Per-device precision and batch size live in `clay.profiles` (cuda: bfloat16, batch 32). Only the Clay encoder uses the GPU;
the small B/C heads run on CPU (`models.head_device`).
```powershell
# one-time (PowerShell, repo root; needs micromamba, git and a current NVIDIA driver)
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1   # env + CUDA torch + TerraTorch + weights + GPU check
micromamba activate dct-ml

python -m dct.env_check --clay --probe-batch    # GPU check; Clay fp32 vs bf16 parity; best batch size
python -m dct.us.s2_extract --hub us_nova --limit 5 --device cuda
python -m streamlit run dct/labels/labeler_app.py
```
RTX 50 series (Blackwell) needs a torch build for CUDA 12.8+ (`cu128`); the setup script installs it from the PyTorch
index and re-checks that `torch.version.cuda` is not `None` after TerraTorch is installed. Earthdata login for OPERA:
set `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`, or create `%USERPROFILE%\_netrc`. Update later with `scripts/sync.ps1`.
Large folders (`models/`, `data/raw/`, embeddings) are not in git; copy between machines with robocopy/rsync.

### Environment (ML, Mac)
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
python -m streamlit run dct/labels/labeler_app.py                  # label, save (Mac/Linux alt: PYTHONPATH=. streamlit run ...)
python -m dct.clay.autolabel_naip                                  # trains, writes anchors_naip.csv + next queue

# 5 monthly labels (stages only go forward)
python -m dct.labels.build_monthly

# 6 Sentinel-2 features + Clay embeddings (use --no-clay for features only), OPERA radar
python -m dct.us.s2_extract
python -m dct.us.opera_rtc                                          # needs an Earthdata login (env vars or netrc)

# 7 models A / B / C, leave-one-hub-out + temporal hold-out, and the release bundle
python -m dct.benchmark.evaluate
python -m dct.benchmark.build_release
```
Smoke tests first: add `--hub us_nova --limit 5` to steps 3, 4 (naip_embed), 6.
