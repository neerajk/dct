"""US benchmark step 5: anchors -> monthly soft stage labels (left-to-right HMM).

Run:  python -m dct.labels.build_monthly
Reads  data/processed/us/anchors_epoch.csv and anchors_naip.csv (either may be missing).
Writes data/processed/us/monthly_labels.csv   campus_id, month, p0..p4, stage (Viterbi path), weight, anchored
       outputs/us/label_conflicts.csv         anchors the final labels disagree with (for review)
       outputs/us/monthly_labels_summary.csv  stage counts

Model. Hidden state = stage 0..4 of a campus in a month. Transitions are left-to-right: stay with
labels.hmm.stay_prob, advance one stage with advance_prob, skip one with skip_prob, NEVER go back;
stage 4 is absorbing. An anchor (stage s, date d, uncertainty u months, reliability r) is soft evidence
for the months within +/- u of d: strength = min(r, 0.98) * (1 - |month - d| / (u + 1)), and the
likelihood of stage k is strength * [k == s] + (1 - strength) / K (so no anchor is ever 'impossible' and
conflicts are settled by reliability: manual > NAIP-auto > Epoch). Forward-backward gives the monthly
stage probabilities; weight = max(posterior, weight_floor) so months near a transition or far from any
anchor count less in training. The Viterbi path is asserted never to go backward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dct.common import load_us_config, month_starts, out_dir, proc_dir

K = 5


def transition_matrix(h: dict) -> np.ndarray:
    A = np.zeros((K, K))
    for k in range(K):
        A[k, k] = h["stay_prob"]
        if k + 1 < K:
            A[k, k + 1] = h["advance_prob"]
        if k + 2 < K:
            A[k, k + 2] = h["skip_prob"]
        if k == K - 1:
            A[k, k] = 1.0
        A[k] /= A[k].sum()
    return A


def forward_backward(L: np.ndarray, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    T = len(L)
    alpha = np.zeros((T, K))
    a = pi * L[0]
    alpha[0] = a / a.sum()
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * L[t]
        alpha[t] = a / a.sum()
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        b = A @ (L[t + 1] * beta[t + 1])
        beta[t] = b / b.sum()
    g = alpha * beta
    return g / g.sum(1, keepdims=True)


def viterbi(L: np.ndarray, A: np.ndarray, pi: np.ndarray) -> np.ndarray:
    T = len(L)
    logA = np.log(np.where(A > 0, A, 1e-300))
    delta = np.zeros((T, K))
    back = np.zeros((T, K), dtype=int)
    delta[0] = np.log(pi) + np.log(L[0])
    for t in range(1, T):
        cand = delta[t - 1][:, None] + logA
        back[t] = cand.argmax(0)
        delta[t] = cand.max(0) + np.log(L[t])
    path = np.zeros(T, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def likelihoods(months: list[pd.Timestamp], anchors: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(T, K) soft-evidence likelihoods and a (T,) bool 'has anchor evidence' mask."""
    T = len(months)
    L = np.ones((T, K))
    anchored = np.zeros(T, dtype=bool)
    mi = {m.strftime("%Y-%m"): i for i, m in enumerate(months)}
    for a in anchors.itertuples():
        d = pd.Timestamp(a.date)
        u = int(a.uncertainty_months)
        for off in range(-u, u + 1):
            m = (d.replace(day=1) + pd.DateOffset(months=off)).strftime("%Y-%m")
            if m not in mi:
                continue
            strength = min(float(a.reliability), 0.98) * (1 - abs(off) / (u + 1))
            v = np.full(K, (1 - strength) / K)
            v[int(a.stage)] += strength
            L[mi[m]] *= v
            L[mi[m]] /= L[mi[m]].sum()
            anchored[mi[m]] = True
    return L, anchored


def load_anchors(cfg: dict) -> pd.DataFrame:
    parts = []
    for name in ("anchors_epoch.csv", "anchors_naip.csv"):
        p = proc_dir(cfg, name)
        if p.exists():
            df = pd.read_csv(p)
            keep = ["campus_id", "date", "stage", "uncertainty_months", "source", "reliability"]
            parts.append(df[keep])
    if not parts:
        raise SystemExit("no anchors: run dct.us.epoch_import and/or dct.clay.autolabel_naip first")
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        raise SystemExit("anchor files exist but contain no anchors: check outputs/us/epoch_unmapped.csv, "
                         "epoch_matches.csv and epoch_geocode_failed.csv, and label some NAIP campus-years")
    return out


def main() -> None:
    cfg = load_us_config()
    lcfg = cfg["us"]["labels"]
    anchors = load_anchors(cfg)
    A = transition_matrix(lcfg["hmm"])
    pi = np.full(K, 1.0 / K)
    start = cfg["time"]["start_floor"]
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    months = month_starts(start, end)

    rows, conflicts, n_flicker = [], [], 0
    for cid, grp in anchors.groupby("campus_id"):
        L, anchored = likelihoods(months, grp)
        gamma = forward_backward(L, A, pi)
        path = viterbi(L, A, pi)
        assert (np.diff(path) >= 0).all(), f"{cid}: Viterbi path goes backward (bug)"
        n_flicker += int((np.diff(gamma.argmax(1)) < 0).any())
        weight = np.maximum(gamma.max(1), lcfg["weight_floor"])
        for t, m in enumerate(months):
            rows.append({"campus_id": cid, "month": m.strftime("%Y-%m-%d"), **{f"p{k}": gamma[t, k] for k in range(K)},
                         "stage": int(path[t]), "weight": float(weight[t]), "anchored": bool(anchored[t])})
        for a in grp.itertuples():
            m = pd.Timestamp(a.date).replace(day=1).strftime("%Y-%m-%d")
            t = next((i for i, mm in enumerate(months) if mm.strftime("%Y-%m-%d") == m), None)
            if t is not None and path[t] != int(a.stage):
                conflicts.append({"campus_id": cid, "date": a.date, "anchor_stage": int(a.stage), "source": a.source,
                                  "reliability": a.reliability, "final_stage": int(path[t]),
                                  "p_anchor_stage": float(gamma[t, int(a.stage)])})
    df = pd.DataFrame(rows)
    p, o = proc_dir(cfg), out_dir(cfg)
    p.mkdir(parents=True, exist_ok=True)
    o.mkdir(parents=True, exist_ok=True)
    df.to_csv(p / "monthly_labels.csv", index=False)
    pd.DataFrame(conflicts).to_csv(o / "label_conflicts.csv", index=False)
    summ = df.groupby("stage").size().rename("campus_months").reset_index()
    summ.to_csv(o / "monthly_labels_summary.csv", index=False)
    print(f"{df['campus_id'].nunique()} campuses x {len(months)} months -> {p / 'monthly_labels.csv'}")
    print(f"{len(conflicts)} anchor(s) overruled by other evidence -> label_conflicts.csv")
    print(f"campuses whose marginal argmax flickers backward (Viterbi path never does): {n_flicker}")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
