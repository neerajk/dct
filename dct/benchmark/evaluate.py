"""Benchmark runner: models A (LightGBM baseline), B (Clay tracker), C (Clay tracker + NAIP teacher).

Run (ML env, after the pipeline steps):  python -m dct.benchmark.evaluate [--models A B C] [--folds hub temporal]
Folds
  hub       leave-one-hub-out: train on four hubs, test on the fifth (proxy for a new region)
  temporal  train on months before the last 12, test on the last 12 (all campuses)
Scored months = months with an image, a label and a reliable reference (label weight >= 0.9: a confident posterior).
Two subsets: 'all' and 'no_naip' (months NOT paired with a NAIP flight) - the honest test of model C, because
those are months where the teacher gave no direct hint.
Metrics per model/fold/subset: macro-F1, within-1-stage accuracy, MAE in stages, Brier score of the smoothed
stage probabilities, and (hub folds, subset 'all' only) the anchor-interval transition error: for each pair of
consecutive anchors with a stage increase, how many months the predicted path reaches the new stage outside the
interval between the two anchors (0 = inside), plus the share of transitions never detected (miss rate).
All models are HMM-smoothed (labels.hmm) so predictions never go backward.
Outputs: outputs/us/benchmark_scores.csv, outputs/us/oof_<model>.npy + oof_index.csv (out-of-fold smoothed posteriors).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from dct.benchmark.dataset import Bench, assemble
from dct.common import load_us_config, out_dir
from dct.models.smooth import smooth_all

K = 5


def interval_transition_error(anchors: dict, path: np.ndarray, test_n: np.ndarray, T: int) -> tuple[float, float, int]:
    """Anchor-interval transition error. For two consecutive anchors (month ta stage sa, month tb stage sb > sa) the
    change to each stage s in sa+1..sb happened somewhere in (ta, tb). The error of a predicted path is 0 if it first
    reaches s inside [ta, tb], else the distance in months to the nearest end; if it never reaches s the error is
    the months from tb to the end of the series and the case counts as a miss.
    Returns (mean error in months, miss rate, number of evaluated transitions)."""
    errs, misses = [], 0
    for row, n in enumerate(test_n):
        a = anchors.get(int(n), [])
        for (ta, sa), (tb, sb) in zip(a, a[1:]):
            if sb <= sa or tb <= ta:
                continue
            for s in range(sa + 1, sb + 1):
                reached = np.where(path[row] >= s)[0]
                if len(reached) == 0:
                    misses += 1
                    errs.append(max(T - 1 - tb, 1))
                    continue
                tp = int(reached[0])
                errs.append(0 if ta <= tp <= tb else min(abs(tp - ta), abs(tp - tb)))
    if not errs:
        return float("nan"), float("nan"), 0
    return float(np.mean(errs)), misses / len(errs), len(errs)


def score(ref: np.ndarray, post: np.ndarray, path: np.ndarray, mask: np.ndarray) -> dict:
    """ref (n, T) reference stages (-1 = none); mask (n, T) months to score."""
    if not mask.any():
        return {"n_months": 0}
    y, p, P = ref[mask], path[mask], post[mask]
    onehot = np.eye(K)[y]
    return {"n_months": int(mask.sum()), "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "within1": float((np.abs(p - y) <= 1).mean()), "mae_stage": float(np.abs(p - y).mean()),
            "brier": float(((P - onehot) ** 2).sum(1).mean())}


def run_models(cfg: dict, b: Bench, train_n: np.ndarray, test_n: np.ndarray, models: list[str],
               cutoff: int | None, device: str) -> dict[str, np.ndarray]:
    mcfg = cfg["us"]["models"]
    out: dict[str, np.ndarray] = {}
    if "A" in models:
        from dct.models.baseline_lgbm import fit_predict
        out["A"] = fit_predict(b, train_n, test_n, mcfg["baseline_lgbm"], cutoff)
    if "B" in models or "C" in models:
        from dct.clay.tracker_s2 import predict_seq, train_seq
        if b.X is None:
            raise SystemExit("models B/C need Clay embeddings: assemble(need_clay=True)")
        W = b.W.copy()
        if cutoff is not None:
            W[:, cutoff:] = 0
        mask = np.zeros(len(b.campuses), bool)
        mask[train_n] = True
        if "B" in models:
            m, sc = train_seq(b.X, b.M, b.Q, W, mcfg["clay_head"], mcfg["seed"], device, train_mask=mask, cutoff=cutoff)
            out["B"] = predict_seq(m, sc, b.X[test_n], b.M[test_n], device)
        if "C" in models:
            TM = b.TM.copy()
            if cutoff is not None:
                TM[:, cutoff:] = False
            if not TM.any():
                print("WARNING: no NAIP-paired months: model C is identical to B (run autolabel_naip first)")
            m, sc = train_seq(b.X, b.M, b.Q, W, mcfg["clay_head"], mcfg["seed"], device, TQ=b.TQ, TM=TM,
                              dcfg=mcfg["distill"], train_mask=mask, cutoff=cutoff)
            out["C"] = predict_seq(m, sc, b.X[test_n], b.M[test_n], device)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    ap.add_argument("--folds", nargs="+", default=["hub", "temporal"], choices=["hub", "temporal"])
    args = ap.parse_args()
    cfg = load_us_config()
    need_clay = any(m in args.models for m in ("B", "C"))
    b = assemble(cfg, need_clay=need_clay, need_teacher="C" in args.models)
    N, T = len(b.campuses), len(b.months)
    print(f"{N} labelled campuses x {T} months; {int(b.M.sum())} campus-months with an S2 image")
    device = "cpu"
    if need_clay:
        from dct.clay.encoder import pick_device
        device = str(pick_device(cfg["us"]["clay"].get("device")))
    hmm = cfg["us"]["labels"]["hmm"]
    hold = cfg["us"]["models"]["eval"]["temporal_holdout_months"]

    rows = []
    oof = {m: np.full((N, T, K), np.nan, dtype="float32") for m in args.models}

    def evaluate_fold(fold: str, test_n: np.ndarray, probs: dict, lo: int = 0):
        for mname, P in probs.items():
            post, path = smooth_all(P, hmm)
            sub_ref = b.REF[test_n]
            base = b.M[test_n] & b.RELIABLE[test_n] & (sub_ref >= 0)
            base[:, :lo] = False
            subsets = {"all": base}
            if b.TM is not None:
                subsets["no_naip"] = base & ~b.TM[test_n]
            if fold != "temporal":
                t_err, t_miss, t_n = interval_transition_error(b.anchors, path, test_n, T)
            else:
                t_err = t_miss = float("nan"); t_n = 0
            for sname, mask in subsets.items():
                extra = {"transition_err_months": t_err, "transition_miss_rate": t_miss, "n_transitions": t_n} if sname == "all" else {}
                rows.append({"model": mname, "fold": fold, "subset": sname, **score(sub_ref, post, path, mask), **extra})
            if fold != "temporal":
                oof[mname][test_n] = post

    if "hub" in args.folds:
        for h in sorted(set(b.hubs)):
            test_n = np.where(b.hubs == h)[0]
            train_n = np.where(b.hubs != h)[0]
            if len(train_n) < 10 or len(test_n) < 3:
                print(f"skip hub {h}: too few campuses (train {len(train_n)}, test {len(test_n)})")
                continue
            print(f"fold hub={h}: train {len(train_n)} test {len(test_n)}")
            evaluate_fold(f"hub:{h}", test_n, run_models(cfg, b, train_n, test_n, args.models, None, device))
    if "temporal" in args.folds:
        cutoff = T - hold
        all_n = np.arange(N)
        print(f"fold temporal: train months < {b.months[cutoff]}, test last {hold} months")
        evaluate_fold("temporal", all_n, run_models(cfg, b, all_n, all_n, args.models, cutoff, device), lo=cutoff)

    res = pd.DataFrame(rows)
    o = out_dir(cfg)
    o.mkdir(parents=True, exist_ok=True)
    res.to_csv(o / "benchmark_scores.csv", index=False)
    pd.DataFrame({"campus_id": b.campuses, "hub": b.hubs}).to_csv(o / "oof_index.csv", index=False)
    for m, a in oof.items():
        np.save(o / f"oof_{m}.npy", a)
    metric_cols = [c for c in ("macro_f1", "within1", "mae_stage", "brier", "transition_err_months", "transition_miss_rate") if c in res]
    if len(res) and "macro_f1" in res:
        summary = res.groupby(["model", "subset"])[metric_cols].mean().round(3)
        print("\nMean over folds (hub folds + temporal; transition columns: hub folds, subset all):")
        print(summary.to_string())
    else:
        print("No scored months in any fold (need months with an image, a label and a reliable reference).")
    print(f"\nPer-fold scores: {o / 'benchmark_scores.csv'}")


if __name__ == "__main__":
    main()
