"""US benchmark step 4b: the NAIP TEACHER - Clay v1.5 embeddings + a small ordinal head label every NAIP campus-year.

Run (ML env, after dct.clay.naip_embed):  python -m dct.clay.autolabel_naip
Loop:
  1. First run (fewer than autolabel.min_labels_to_train manual labels): writes
     data/processed/us/label_queue.csv = a hub-balanced random sample of campus-years to label by hand.
  2. Label them in the app:  PYTHONPATH=. streamlit run dct/labels/labeler_app.py   (saves manual_labels.csv)
  3. Run this again: trains an ensemble of ordinal heads on the manual labels, reports leave-one-hub-out
     scores, predicts every campus-year with a confidence, and writes the next label_queue.csv =
     the least-confident unlabeled campus-years (active learning). Repeat until macro-F1 stops improving.
Outputs (data/processed/us/):
  naip_predictions.csv  campus_id, year, date, hub, stage_pred, conf, p0..p4, labelled, excluded
  anchors_naip.csv      manual labels (reliability 1.0) + auto predictions with conf >= review threshold
  label_queue.csv       what to label next
Frozen encoder + head (not TerraTorch's ClassificationTask): the stock Clay1_5ModelFactory feeds zero
time/location, so DCT pools its own embeddings (naip_embed) and trains only the head here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from dct.clay.heads import fit_head, pooled_features, predict_probs
from dct.clay.store import load_all
from dct.common import load_us_config, proc_dir

K = 5


def load_manual(cfg: dict) -> pd.DataFrame:
    p = proc_dir(cfg, "manual_labels.csv")
    if not p.exists():
        return pd.DataFrame(columns=["campus_id", "year", "stage"])
    df = pd.read_csv(p)
    return df.sort_values("timestamp").drop_duplicates(["campus_id", "year"], keep="last") if "timestamp" in df else df


def split_manual(man: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """(labels with stage >= 0, set of (campus_id, year) the labeller marked unusable = stage -1)."""
    skip = set(map(tuple, man.loc[man["stage"] < 0, ["campus_id", "year"]].to_numpy()))
    return man[man["stage"] >= 0], skip


def onehot(stage: np.ndarray) -> np.ndarray:
    q = np.zeros((len(stage), K), dtype="float32")
    q[np.arange(len(stage)), stage.astype(int)] = 1.0
    return q


def seed_queue(idx: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    hubs = sorted(idx["hub"].unique())
    per = int(np.ceil(n / max(len(hubs), 1)))
    for h in hubs:
        sub = idx[idx["hub"] == h]
        parts.append(sub.iloc[rng.permutation(len(sub))[:per]])
    return pd.concat(parts).head(n)[["campus_id", "year", "hub"]]


def active_queue(pred: pd.DataFrame, n: int) -> pd.DataFrame:
    """Least-confident unlabeled rows, taken round-robin across hubs."""
    un = pred[~pred["labelled"] & ~pred["excluded"]].sort_values("conf")
    groups = [g.reset_index(drop=True) for _, g in un.groupby("hub")]
    rows = []
    for i in range(max((len(g) for g in groups), default=0)):
        for g in groups:
            if i < len(g):
                rows.append(g.iloc[i])
    if not rows:
        return pd.DataFrame(columns=["campus_id", "year", "hub", "conf"])
    return pd.DataFrame(rows).head(n)[["campus_id", "year", "hub", "conf"]]


def main() -> None:
    cfg = load_us_config()
    acfg = cfg["us"]["autolabel"]
    idx, arr = load_all(proc_dir(cfg, "naip_emb"))
    if idx.empty:
        raise SystemExit("no NAIP embeddings: run python -m dct.us.naip_chips then python -m dct.clay.naip_embed")
    X = pooled_features(arr)
    man, skip = split_manual(load_manual(cfg))
    key = idx[["campus_id", "year"]].merge(man[["campus_id", "year", "stage"]], on=["campus_id", "year"], how="left")
    labelled = key["stage"].notna().to_numpy()
    excluded = np.array([(c, y) in skip for c, y in zip(idx["campus_id"], idx["year"])])
    print(f"{len(idx)} NAIP campus-years, {int(labelled.sum())} with manual labels, {int(excluded.sum())} marked unusable")
    p = proc_dir(cfg)
    p.mkdir(parents=True, exist_ok=True)

    if labelled.sum() < acfg["min_labels_to_train"]:
        pool = idx[~labelled & ~excluded]
        if pool.empty:
            pd.DataFrame(columns=["campus_id", "year", "hub"]).to_csv(p / "label_queue.csv", index=False)
            raise SystemExit(f"No unlabelled campus-years left and only {int(labelled.sum())} manual labels "
                             f"(< autolabel.min_labels_to_train={acfg['min_labels_to_train']}). Run dct.us.naip_chips "
                             "without --limit for more campuses, or lower autolabel.min_labels_to_train.")
        q = seed_queue(pool, acfg["seed_labels_target"], acfg["seed"])
        q.to_csv(p / "label_queue.csv", index=False)
        print(f"Too few labels to train (< {acfg['min_labels_to_train']}). Wrote {len(q)} campus-years to label_queue.csv.")
        print("Label them:  PYTHONPATH=. streamlit run dct/labels/labeler_app.py")
        return

    y = key["stage"].to_numpy()
    hub = idx["hub"].to_numpy()
    Xl, yl = X[labelled], y[labelled].astype(int)

    # --- leave-one-hub-out check on the manual labels
    for h in sorted(set(hub[labelled])):
        te = hub[labelled] == h
        if te.sum() < 5 or (~te).sum() < acfg["min_labels_to_train"]:
            continue
        model, sc = fit_head(Xl[~te], onehot(yl[~te]), None, acfg["head"], K, acfg["seed"])
        pred = predict_probs(model, sc, Xl[te]).argmax(1)
        print(f"  hold out {h}: n={int(te.sum())}  macro-F1={f1_score(yl[te], pred, average='macro'):.3f}  "
              f"MAE={np.abs(pred - yl[te]).mean():.2f} stages")

    # --- final ensemble on all labels -> predict every campus-year
    probs = []
    for s in range(acfg["ensemble"]):
        model, sc = fit_head(Xl, onehot(yl), None, acfg["head"], K, acfg["seed"] + s)
        probs.append(predict_probs(model, sc, X))
    P = np.mean(probs, axis=0)
    approx = idx["date_approx"].fillna(False).astype(bool).to_numpy() if "date_approx" in idx else np.zeros(len(idx), bool)
    pred = pd.DataFrame({"campus_id": idx["campus_id"], "year": idx["year"], "date": idx["date"], "hub": idx["hub"],
                         "date_approx": approx,
                         "stage_pred": P.argmax(1), "conf": P.max(1), "labelled": labelled, "excluded": excluded})
    for k in range(K):
        pred[f"p{k}"] = P[:, k]
    pred.to_csv(p / "naip_predictions.csv", index=False)

    # --- anchors: manual first, then confident auto predictions
    rel = cfg["us"]["labels"]["source_reliability"]
    unc = cfg["us"]["labels"]["naip_date_uncertainty_months"]
    unc_approx = cfg["us"]["labels"]["naip_approx_date_uncertainty_months"]
    anchors = []
    for i, r in pred.iterrows():
        if r["excluded"]:
            continue                                   # labeller marked this image unusable: no anchor at all
        if r["labelled"]:
            anchors.append({"campus_id": r["campus_id"], "date": r["date"], "stage": int(y[i]),
                            "uncertainty_months": unc_approx if r["date_approx"] else unc, "source": "manual", "reliability": rel["manual"], "conf": 1.0})
        elif r["conf"] >= acfg["review_confidence_below"]:
            anchors.append({"campus_id": r["campus_id"], "date": r["date"], "stage": int(r["stage_pred"]),
                            "uncertainty_months": unc_approx if r["date_approx"] else unc, "source": "naip_auto", "reliability": rel["naip_auto"],
                            "conf": float(r["conf"])})
    a = pd.DataFrame(anchors)
    a.insert(0, "anchor_id", [f"NAIP_{i:05d}" for i in range(1, len(a) + 1)])
    a.to_csv(p / "anchors_naip.csv", index=False)

    q = active_queue(pred, acfg["per_round_new_labels"])
    q.to_csv(p / "label_queue.csv", index=False)
    low = int((~pred["labelled"] & ~pred["excluded"] & (pred["conf"] < acfg["review_confidence_below"])).sum())
    print(f"anchors_naip.csv: {len(a)} anchors ({int((a['source'] == 'manual').sum())} manual, "
          f"{int((a['source'] == 'naip_auto').sum())} auto). {low} low-confidence campus-years left for review.")
    print(f"label_queue.csv: the {len(q)} least-confident campus-years. Label them, then run this again.")


if __name__ == "__main__":
    main()
