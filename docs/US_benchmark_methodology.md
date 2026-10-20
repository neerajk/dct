# DCT-US benchmark v0: data flow and methodology

Written for understanding, not for a paper. Reference codes like [D-6] point to `DCT_references.docx`.

## The goal in one paragraph
Build a US dataset in which every data-center campus has a **construction stage for every month**,
then train models that read **free satellite images (Sentinel-2, optionally radar)** and predict that
stage. The same models should later work in India and the EU, where sharp imagery and labels are scarce.

## Why the US first
Three free label sources exist there: building outlines from OpenStreetMap, dated construction facts
from Epoch AI (CC-BY) [D-1], and NAIP aerial photos at 0.3-1 m for 2010-2026 [D-3, D-6]. India has none of these.

## Vocabulary
- **Campus / phase:** a campus is one site; each building is a phase.
- **Stage:** 0 greenfield, 1 cleared/earthworks, 2 structure rising, 3 enclosed, 4 operational.
- **Anchor:** one dated fact about a campus, e.g. "NAIP photo of 2021-07 shows a finished roof = stage 3".
- **Embedding:** the list of numbers a model like Clay produces to summarise an image.
- **Teacher / student:** the teacher sees sharp NAIP images and knows the stage well. The student sees only
  10 m Sentinel-2 images and learns to copy the teacher.
- **HMM (hidden Markov model):** a way to find the most likely stage sequence when stages may only stay or move forward.

## Data flow
```
OSM (5 hubs) ----------> campuses + phases            (data/seed_us/, source=osm, ODbL)
Epoch CSVs (CC-BY) ----> anchors (text -> stage)       (+/- 6 months)
NAIP per flight year --> 1 m chips per campus
        |  ~200 hand labels (Streamlit app)
        v
Clay v1.5 on NAIP + small head  = TEACHER ---------> NAIP anchors (+ confidence, review queue)
        v
anchors (Epoch + NAIP + manual) --HMM (never backward)--> monthly soft labels per campus
        |
Sentinel-2 monthly chips --> Clay v1.5 embeddings (inside footprint + surrounding ring)
OPERA radar + S2 indices --> hand-made features
        |
   A  baseline  : LightGBM on hand-made features
   B  Clay track: ordinal head + GRU on monthly embeddings
   C  student   : B trained also on the teacher's soft targets
        |
   leave-one-hub-out scores  ->  DCT-US v0 release (sites, anchors, labels, scene IDs, splits)
```

## Step by step

### 1. Sites (where)
Overpass queries on OSM for `telecom=data_center`, `building=data_center` and `construction=data_center`
inside five hub boxes (Northern Virginia, Phoenix, Dallas-Fort Worth, Columbus, Atlanta). Buildings in one
campus polygon, or within 150 m of each other, form one campus. Footprints under 2,000 m² are dropped
(under ~20 Sentinel-2 pixels). Rows are tagged `source=osm, licence=ODbL`.

### 2. Anchors (when)
- **Epoch:** Epoch publishes no coordinates, so each US site is geocoded from its Address (Nominatim, cached; you can
  override with `epoch_site_coords.csv`) and matched to the nearest OSM campuses (listed in `epoch_matches.csv` for
  review). The "Construction status" text is mapped to a stage by whole-word keywords after URLs and negated phrases
  ("not fully operational") are removed. The numeric "Buildings operational" column beats the prose: >= 1 means stage 4,
  0 caps the stage at 3. **Campus stage = the most advanced building.** Each row is an anchor with +/-6 months of
  uncertainty (Epoch says 80% of its dates are within 6 months [D-1]). Unmatched text goes to a review file.
- **NAIP:** one stage per campus per flight year. Labelled by the Clay teacher after about 200 manual labels.
  Active learning: label the 50 least-confident chips, retrain, repeat.
- **Conflicts:** manual beats NAIP-auto beats Epoch, and each conflict is logged.

### 3. Few anchors become monthly labels
A site has about 60 monthly images but only about 6 known facts. Because **stages never go backward**:
- between two anchors with the **same** stage, every month gets that stage (certain);
- between **different** stages, the change happened somewhere in the gap, so months get soft probabilities
  from a left-to-right HMM, with lower training weight near the transition [D-10].
Random label noise of 25-30% hurts Random Forest little [D-13], so this is acceptable; down-weighting uncertain labels reduces harm further [D-14].

### 4. Clay v1.5 through TerraTorch
Clay is a satellite foundation model whose patch embedding is conditioned on each band's **wavelength**
and the image's **pixel size (GSD)**, so one model reads NAIP, Sentinel-2 and others [F-1, F-3]. TerraTorch's
`Clay1_5ModelFactory` loads it [F-4], but its wrapper passes **zero time and location**. DCT therefore wraps
TerraTorch's own `Encoder` and supplies real time and lat/lon, with masking switched off so every patch is kept.
The weights live at `models/clay/clay-v1.5.ckpt` (Apache-2.0).

### 4b. Sentinel-2 chips: clouds and offsets
Each month the clearest scene is chosen by the cloud mask (SCL) **over the campus footprint and its ring**, not just
the whole 2.56 km chip (a small cloud on the building would otherwise pass). Months failing either test are gaps. Network
errors are retried and then reported as failures, never recorded as cloud. Cloud/shadow pixels are left out of the
hand-made features. No reflectance-offset correction is applied: reading live Earth Search pixels showed the archive
is already corrected, whatever the `boa_offset_applied` flag says.

### 5. Three models
- **A baseline** (explainable): indices from Sentinel-2 (vegetation, bare soil, built-up), radar brightness (OPERA),
  all as footprint minus surrounding ring, fed to LightGBM, then the same HMM smoothing.
- **B Clay tracker:** monthly Sentinel-2 embeddings, an ordinal head (predicts "stage > k", keeping the order [D-12]),
  a small GRU across months, then the HMM.
- **C student:** model B trained with the labels **and** the teacher's soft probabilities where a NAIP date lines up
  with a Sentinel-2 month (within 30 days). The idea follows 50 cm-teacher to Sentinel-2-student work [F-8] and
  soft-target distillation [F-11].

### 6. What counts as success
All scores use **leave-one-hub-out** (train on four hubs, test on the fifth), as a proxy for a new region.
Per-stage F1, macro-F1, within-1-stage accuracy and the Brier score of the stage probabilities. Transitions are scored
against the **anchors**: for two consecutive anchors with a stage increase, the predicted change must fall between them
(error 0 inside, otherwise the months outside), and the share of transitions a model never detects is reported.
The key question for C: does it beat B on hubs and dates **without NAIP**? If yes, sharp-image knowledge transferred.
Foundation models do not always beat supervised baselines [F-6, F-7], so A is always reported next to B and C.

## Rules the code follows
- NAIP and Esri imagery are **never model inputs** (India has no NAIP). NAIP only makes labels and the teacher.
- Every row carries its source and licence.
- Nothing is silently changed: label conflicts, unmapped Epoch texts and stage-order violations are written to files.
- Neeraj runs all jobs; the code only reports what it did.
