"""Assemble the aligned benchmark arrays from the pipeline outputs.

All campuses share one monthly axis (time.start_floor .. the last month in monthly_labels.csv, so results do not
change when you re-run in a later calendar month), so a campus is an (T, ...) sequence:
  Q    (N, T, 5)  soft stage labels from build_monthly          W (N, T) training weight (0 = no label or no image)
  REF  (N, T)     Viterbi reference stage (-1 = none)            RELIABLE (N, T) months to score: label weight >= 0.9
  M    (N, T)     True where a usable Sentinel-2 chip exists (and, with Clay, an embedding for it)
  X    (N, T, F)  pooled Clay features (inside, inside - ring); zeros where M is False   [need_clay]
  TQ   (N, T, 5)  teacher stage distribution for months within pair_window_days of a NAIP flight; TM its mask [need_teacher]
  anchors  {campus index: [(month index, stage), ...]} sorted by month, for the anchor-interval transition metric
  feats DataFrame  long table of hand-made S2 + OPERA features with change features, for the LightGBM baseline
Only campuses that have monthly labels are included.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dct.clay.heads import pooled_features
from dct.clay.store import load_all
from dct.common import load_us_config, month_starts, proc_dir, read_seed

K = 5
CHANGE_COLS = ["ndvi_diff", "ndbi_diff", "bsi_diff", "bright_diff", "swir2_diff", "ndvi_in", "bright_in",
               "vv_diff", "vh_diff", "vv_in", "vh_in"]


@dataclass
class Bench:
    campuses: list[str]
    hubs: np.ndarray
    months: list[str]
    Q: np.ndarray
    W: np.ndarray
    REF: np.ndarray
    RELIABLE: np.ndarray
    M: np.ndarray
    X: np.ndarray | None = None
    TQ: np.ndarray | None = None
    TM: np.ndarray | None = None
    anchors: dict = field(default_factory=dict)
    feats: pd.DataFrame = field(default_factory=pd.DataFrame)


def month_keys(cfg: dict) -> list[str]:
    """Monthly axis from the start floor to the last month present in monthly_labels.csv (data-derived)."""
    lab = pd.read_csv(proc_dir(cfg, "monthly_labels.csv"), usecols=["month"])
    end = pd.to_datetime(lab["month"]).max().strftime("%Y-%m-%d")
    return [m.strftime("%Y-%m") for m in month_starts(cfg["time"]["start_floor"], end)]


def _concat_csv(folder, with_campus: bool = True) -> pd.DataFrame:
    parts = []
    for p in sorted(folder.glob("*.csv")):
        df = pd.read_csv(p)
        if len(df):
            if with_campus:
                df.insert(0, "campus_id", p.stem)
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _change_features(df: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """Add <col>_rel0 (minus the median of the campus's first 6 available observations) and <col>_d3 (change vs
    exactly 3 months earlier; NaN if that month had no image). Computed on the full monthly axis so a cloud gap
    never turns '3 months' into 5."""
    out = []
    for cid, g in df.drop_duplicates(["campus_id", "month"]).groupby("campus_id"):
        g = g.set_index("month").reindex(months)
        g.index.name = "month"
        g["campus_id"] = cid
        for c in CHANGE_COLS:
            if c in g:
                first = g[c].dropna().iloc[:6]
                g[f"{c}_rel0"] = g[c] - (first.median() if len(first) else np.nan)
                g[f"{c}_d3"] = g[c] - g[c].shift(3)
        out.append(g[g["status"] == "ok"].reset_index())
    return pd.concat(out, ignore_index=True) if out else df


def _anchors(cfg: dict, ci: dict, mi: dict) -> dict[int, list[tuple[int, int]]]:
    res: dict[int, list[tuple[int, int]]] = {}
    for name in ("anchors_epoch.csv", "anchors_naip.csv"):
        p = proc_dir(cfg, name)
        if not p.exists():
            continue
        a = pd.read_csv(p)
        for r in a.itertuples():
            if r.campus_id in ci and r.reliability >= 0.5:
                m = pd.Timestamp(r.date).strftime("%Y-%m")
                if m in mi:
                    res.setdefault(ci[r.campus_id], []).append((mi[m], int(r.stage)))
    return {n: sorted(v) for n, v in res.items()}


def assemble(cfg: dict | None = None, need_clay: bool = False, need_teacher: bool = False) -> Bench:
    cfg = cfg or load_us_config()
    months = month_keys(cfg)
    T = len(months)
    mi = {m: i for i, m in enumerate(months)}

    lab = pd.read_csv(proc_dir(cfg, "monthly_labels.csv"))
    lab["month"] = lab["month"].str[:7]
    campuses = sorted(lab["campus_id"].unique())
    ci = {c: i for i, c in enumerate(campuses)}
    N = len(campuses)
    hub_of = dict(zip(*read_seed(cfg, "campuses")[["campus_id", "cluster"]].T.values))
    hubs = np.array([hub_of[c] for c in campuses])

    Q, W = np.zeros((N, T, K), "float32"), np.zeros((N, T), "float32")
    REF, ANCH = -np.ones((N, T), int), np.zeros((N, T), bool)
    rows = lab[lab["month"].isin(list(mi))]
    n_idx = rows["campus_id"].map(ci).to_numpy()
    t_idx = rows["month"].map(mi).to_numpy()
    Q[n_idx, t_idx] = rows[[f"p{k}" for k in range(K)]].to_numpy(dtype="float32")
    W[n_idx, t_idx] = rows["weight"].to_numpy(dtype="float32")
    REF[n_idx, t_idx] = rows["stage"].to_numpy()
    ANCH[n_idx, t_idx] = rows["anchored"].to_numpy()
    RELIABLE = W >= 0.9                       # confident posterior only; months inside a weak anchor's window do not count

    s2 = _concat_csv(proc_dir(cfg, "s2"))
    if len(s2):
        s2 = s2[(s2["status"] == "ok") & s2["campus_id"].isin(list(ci)) & s2["month"].isin(list(mi))]
    M = np.zeros((N, T), bool)
    if len(s2):
        M[s2["campus_id"].map(ci).to_numpy(), s2["month"].map(mi).to_numpy()] = True

    feats = s2.copy()
    opera = _concat_csv(proc_dir(cfg, "opera"))
    if len(opera) and len(feats):
        keep = ["campus_id", "month", "vv_in", "vv_ring", "vv_diff", "vh_in", "vh_ring", "vh_diff"]
        feats = feats.merge(opera[keep], on=["campus_id", "month"], how="left")
    if len(feats):
        feats = _change_features(feats, months)

    b = Bench(campuses, hubs, months, Q, W * M, REF, RELIABLE, M, feats=feats, anchors=_anchors(cfg, ci, mi))

    if need_clay:
        idx, arr = load_all(proc_dir(cfg, "s2_emb"))
        if idx.empty:
            raise SystemExit("no S2 embeddings: run python -m dct.us.s2_extract (without --no-clay)")
        Fe = pooled_features(arr)
        keep = (idx["campus_id"].isin(list(ci)) & idx["month"].isin(list(mi))).to_numpy()
        X = np.zeros((N, T, Fe.shape[1]), "float32")
        en, et = idx["campus_id"][keep].map(ci).to_numpy(), idx["month"][keep].map(mi).to_numpy()
        X[en, et] = Fe[keep]
        has = np.zeros((N, T), bool)
        has[en, et] = True
        if (M & ~has).any():
            print(f"note: {int((M & ~has).sum())} campus-months have an S2 chip but no embedding; "
                  "they are excluded for ALL models so A, B and C are scored on the same months")
        b.M = M & has
        b.W = W * b.M
        b.X = X

    if need_teacher:
        TQ, TM = np.zeros((N, T, K), "float32"), np.zeros((N, T), bool)
        pred_p = proc_dir(cfg, "naip_predictions.csv")
        if pred_p.exists() and len(s2):
            pred = pd.read_csv(pred_p)
            man_p = proc_dir(cfg, "manual_labels.csv")
            man = {}
            if man_p.exists():
                m_df = pd.read_csv(man_p).drop_duplicates(["campus_id", "year"], keep="last")
                man = m_df[m_df["stage"] >= 0].set_index(["campus_id", "year"])["stage"].to_dict()
            win = pd.Timedelta(days=cfg["us"]["models"]["distill"]["pair_window_days"])
            s2d = s2.assign(d=pd.to_datetime(s2["date"]))
            by_c = {c: g for c, g in s2d.groupby("campus_id")}
            for r in pred.itertuples():
                if r.campus_id not in ci or r.campus_id not in by_c or getattr(r, "excluded", False):
                    continue
                g = by_c[r.campus_id]
                gap = (g["d"] - pd.Timestamp(r.date)).abs()
                j = gap.idxmin()
                if gap[j] > win:
                    continue
                n, t = ci[r.campus_id], mi[g.loc[j, "month"]]
                if (r.campus_id, r.year) in man:
                    v = np.zeros(K, "float32")
                    v[int(man[(r.campus_id, r.year)])] = 1.0
                else:
                    v = np.array([getattr(r, f"p{k}") for k in range(K)], "float32")
                TQ[n, t], TM[n, t] = v, True
            TM &= b.M                                   # teacher targets only where the model has an input
        b.TQ, b.TM = TQ, TM
    return b
