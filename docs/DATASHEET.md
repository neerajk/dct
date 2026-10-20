# DCT-US benchmark v0: datasheet (draft)

Status: draft, written before the data exists. Update after each build step.

## Purpose
A benchmark for tracking data-center construction stage from free satellite imagery, built on
five US hubs, so models can be trained where labels are dense and tested for transfer to India and the EU.

## Contents (planned)
| File | What | Licence of the source |
|---|---|---|
| `sites.geojson` | Campus and building outlines for 5 hubs | OpenStreetMap contributors, ODbL (share-alike applies to a published database of these outlines) |
| `anchors.csv` | Dated stage facts: Epoch, NAIP-auto, manual | Epoch AI timelines: CC-BY (credit required). NAIP-derived labels: derived from public-domain imagery. Manual: ours |
| `monthly_labels.parquet` | Soft stage probabilities per campus per month | Derived from the anchors |
| `scenes.csv` | Sentinel-2, OPERA and NAIP item IDs (no imagery redistributed) | Sentinel-2: Copernicus open; OPERA: CC BY 4.0; NAIP: public domain |
| `splits.json` | Leave-one-hub-out and a 12-month temporal hold-out | Ours |

Every row carries `data_source` and `licence`, so the licence of any subset can be decided at publication.

## Stages
0 greenfield, 1 cleared/earthworks, 2 structure rising, 3 enclosed, 4 operational. Stages only move
forward in a campus's history (assumption; violations are logged, not silently fixed).

## Known biases and limits
- **OpenStreetMap completeness varies by area.** Missing outlines mean "not mapped", not "no data center".
- **Mostly finished buildings.** Construction sequences are reconstructed from the imagery archive.
- **Epoch covers frontier AI campuses only** (about 94 sites, 77 in the US). Dates carry about +/-6 months of uncertainty.
- **NAIP is flown every 2-3 years per state**, so anchors are sparse. NAIP stage labels come from a model (Clay v1.5 plus a small head) after about 200 manual labels; low-confidence ones are reviewed.
- **Monthly labels are partly interpolated**, with lower weight near transitions.
- **Hub bounding boxes are approximate** and may miss or include sites at the edges.
- **Transfer is not guaranteed.** Roof materials, monsoon cloud gaps and fewer Sentinel-1 passes in India differ from the US.

## Intended and unintended use
Intended: research on construction-stage tracking and transfer across regions.
Not intended: judging any named company, or any credit decision, from this benchmark alone.

## Maintenance
Rebuilt from sources by script; scene IDs allow re-download. Versions are tagged by date.
