"""NAIP stage labeller (Streamlit). Run from the repo root:

    PYTHONPATH=. streamlit run dct/labels/labeler_app.py

Shows every NAIP year of a campus side by side; pick a stage per year and Save. Labels append to
data/processed/us/manual_labels.csv (campus_id, year, stage, note, labeller, timestamp); the newest
row per campus-year wins. The campus list comes from label_queue.csv (written by
dct.clay.autolabel_naip); without it, every campus that has NAIP chips is listed.

Stages (one image = one campus-year; judge the WHOLE campus footprint, outlined in no way on purpose):
  0 greenfield          farmland, grass, forest, existing unrelated land use
  1 cleared/earthworks  bare soil, graded pads, haul roads, no structures yet
  2 structure rising    slabs, foundations, steel frames, partial walls or roofs
  3 enclosed            full roof and walls; yards (chillers, generators, substation) absent or unfinished
  4 operational         roof + cooling and generator yard + substation, parking in use
Choose 'skip: unusable image' when it is cloudy/black/misaligned (saved as stage -1: the teacher then ignores
that campus-year entirely); leave '-- choose --' for images you have not decided on yet (not saved). Model guesses are hidden by
default so they do not anchor your judgement.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import streamlit as st

from dct.common import load_us_config, proc_dir, raw_dir

STAGES = {-2: "-- choose --", -1: "skip: unusable image", 0: "0 greenfield", 1: "1 cleared / earthworks",
          2: "2 structure rising", 3: "3 enclosed", 4: "4 operational"}


@st.cache_resource
def get_cfg():
    return load_us_config()


def stretch(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a.astype("float32") - lo) / max(hi - lo, 1e-6), 0, 1)


def load_chip(tif: Path, mode: str) -> np.ndarray:
    with rasterio.open(tif) as src:
        d = src.read()
    bands = d[:3] if mode == "RGB" else d[[3, 0, 1]]           # false colour: NIR, red, green
    return np.stack([stretch(b) for b in bands], axis=-1)


def campus_files(cfg, cid: str) -> list[Path]:
    folder = raw_dir(cfg, "naip", cid)
    return sorted(folder.glob("*.tif")) if folder.exists() else []


def read_labels(cfg) -> pd.DataFrame:
    p = proc_dir(cfg, "manual_labels.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["campus_id", "year", "stage", "note", "labeller", "timestamp"])


def save_labels(cfg, rows: list[dict]) -> None:
    p = proc_dir(cfg, "manual_labels.csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    old = read_labels(cfg)
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(p, index=False)


def main() -> None:
    cfg = get_cfg()
    st.set_page_config(page_title="DCT NAIP labeller", layout="wide")
    st.title("DCT: NAIP stage labeller")

    qpath = proc_dir(cfg, "label_queue.csv")
    if qpath.exists() and len(pd.read_csv(qpath)):
        queue = pd.read_csv(qpath)
        campuses = list(dict.fromkeys(queue["campus_id"]))
        st.sidebar.write(f"Queue: {len(queue)} campus-years on {len(campuses)} campuses")
    else:
        campuses = sorted(p.name for p in raw_dir(cfg, "naip").glob("*") if p.is_dir())
        st.sidebar.write(f"No queue file: {len(campuses)} campuses with NAIP chips")
    if not campuses:
        st.warning("No NAIP chips yet. Run python -m dct.us.naip_chips first.")
        return

    labels = read_labels(cfg)
    st.sidebar.write(f"Manual labels so far: {len(labels.drop_duplicates(['campus_id', 'year']))}")
    mode = st.sidebar.radio("Image", ["RGB", "False colour (NIR-R-G)"])
    mode = "RGB" if mode == "RGB" else "CIR"
    min_year = st.sidebar.number_input("Earliest year to show", 2010, 2026, int(cfg["us"]["naip"]["min_year"]))
    labeller = st.sidebar.text_input("Labeller", "neeraj")
    st.sidebar.markdown("**Stages**\n\n0 greenfield\n\n1 cleared / earthworks\n\n2 structure rising\n\n3 enclosed\n\n4 operational")

    if "pos" not in st.session_state:
        st.session_state.pos = 0
    st.session_state.pos = int(np.clip(st.session_state.pos, 0, len(campuses) - 1))
    c1, c2, c3 = st.columns([1, 1, 6])
    if c1.button("Previous"):
        st.session_state.pos = max(0, st.session_state.pos - 1)
        st.rerun()
    if c2.button("Next"):
        st.session_state.pos = min(len(campuses) - 1, st.session_state.pos + 1)
        st.rerun()
    cid = campuses[st.session_state.pos]
    c3.subheader(f"{cid}   ({st.session_state.pos + 1} of {len(campuses)})")

    files = [f for f in campus_files(cfg, cid) if int(f.name.split("_")[0]) >= min_year]
    if not files:
        st.info("No NAIP chip files for this campus in the selected years.")
        return
    cols = st.columns(min(len(files), 4))
    prev = labels[labels["campus_id"] == cid].drop_duplicates("year", keep="last").set_index("year")["stage"].to_dict()
    chosen = {}
    for i, tif in enumerate(files):
        meta = json.loads(tif.with_suffix(".json").read_text())
        col = cols[i % len(cols)]
        col.image(load_chip(tif, mode), caption=f"{meta['year']}  ({meta['date']}{' approx' if meta.get('date_approx') else ''})",
                  use_container_width=True)
        default = int(prev.get(meta["year"], -2))
        chosen[meta["year"]] = col.selectbox("stage", list(STAGES), index=list(STAGES).index(default),
                                             format_func=lambda s: STAGES[s], key=f"{cid}_{meta['year']}")
    note = st.text_input("Note (optional)", key=f"note_{cid}")
    if st.button("Save labels for this campus", type="primary"):
        ts = datetime.now().isoformat(timespec="seconds")
        rows = [{"campus_id": cid, "year": y, "stage": s, "note": note, "labeller": labeller, "timestamp": ts}
                for y, s in chosen.items() if s != -2]          # -1 = explicit skip is saved; -2 = not decided is not
        if rows:
            save_labels(cfg, rows)
            st.success(f"Saved {len(rows)} label(s). Press Next.")
        else:
            st.info("Nothing to save (all still on '-- choose --').")

    if st.sidebar.checkbox("Show model guess for this campus (can anchor you)"):
        pp = proc_dir(cfg, "naip_predictions.csv")
        if pp.exists():
            st.sidebar.dataframe(pd.read_csv(pp).query("campus_id == @cid")[["year", "stage_pred", "conf"]])


if __name__ == "__main__":
    main()
