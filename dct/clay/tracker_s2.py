"""Models B and C: Clay v1.5 Sentinel-2 tracker, and the same tracker distilled from the NAIP teacher.

B  per-month pooled Clay features (inside, inside - ring) -> linear+ReLU -> GRU over the monthly sequence ->
   ordinal (CORAL) head. The GRU is causal (past months only), like a live monitor; HMM smoothing is applied
   afterwards in evaluation (offline, bidirectional).
C  identical network and data, plus a teacher term: in months within pair_window_days of a NAIP flight, the
   output is also pulled toward the NAIP teacher's stage distribution (temperature-softened):
       loss = (1 - alpha) * CORAL(monthly labels, weights) + alpha * CORAL(teacher soft targets)
   so sharp-image knowledge reaches a model that only sees 10 m Sentinel-2 (and so works where NAIP does not exist).
Feature alignment (S2 embedding -> NAIP embedding, cosine) from the plan is left out of v0
(models.distill.feature_align is false and ignored).
Everything is sequence-shaped (N campuses, T months, F), all campuses on one month axis (see benchmark/dataset.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dct.clay.heads import coral_loss, stage_probs


class SeqOrdinal(nn.Module):
    def __init__(self, d_in: int, proj: int = 256, hidden: int = 128, n_stages: int = 5, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in + 1, proj), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(proj, hidden, batch_first=True)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.raw = nn.Parameter(torch.zeros(n_stages - 1))

    def thresholds(self) -> torch.Tensor:
        return self.b0 - torch.cumsum(F.softplus(self.raw), dim=0)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """x (B, T, F) standardised, m (B, T) 1.0 where an image exists -> logits (B, T, K-1)."""
        h = self.inp(torch.cat([x * m[..., None], m[..., None]], dim=-1))
        h, _ = self.gru(h)
        return self.score(h) + self.thresholds()


def soften(q: torch.Tensor, temperature: float) -> torch.Tensor:
    s = q.clamp(min=1e-6) ** (1.0 / temperature)
    return s / s.sum(-1, keepdim=True)


def train_seq(X: np.ndarray, M: np.ndarray, Q: np.ndarray, W: np.ndarray, hcfg: dict, seed: int = 0,
              device: str = "cpu", TQ: np.ndarray | None = None, TM: np.ndarray | None = None,
              dcfg: dict | None = None, train_mask: np.ndarray | None = None, cutoff: int | None = None):
    """Train on the campuses selected by train_mask (bool, N). Returns (model, scaler). Pass TQ/TM/dcfg for model C.
    cutoff = first held-out month index: feature standardisation statistics then use only months before it."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    sel = np.where(train_mask)[0] if train_mask is not None else np.arange(len(X))
    Ms = M[sel].copy()
    if cutoff is not None:
        Ms[:, cutoff:] = False
    Xs = X[sel][Ms]
    mu = Xs.mean(0, keepdims=True)
    sd = Xs.std(0, keepdims=True) + 1e-6
    scaler = {"mu": mu.astype("float32"), "sd": sd.astype("float32")}
    Xn = torch.tensor(((X - mu) / sd) * M[..., None], dtype=torch.float32)
    Mt = torch.tensor(M, dtype=torch.float32)
    Qt = torch.tensor(Q, dtype=torch.float32)
    Wt = torch.tensor(W, dtype=torch.float32)
    use_teacher = TQ is not None and TM is not None and dcfg is not None
    if use_teacher:
        TQt = soften(torch.tensor(TQ, dtype=torch.float32), dcfg["temperature"])
        TMt = torch.tensor(TM, dtype=torch.bool)
        alpha = dcfg["alpha_teacher"]

    model = SeqOrdinal(X.shape[2], hcfg["hidden"], hcfg["gru_hidden"], Q.shape[2], hcfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hcfg["lr"], weight_decay=hcfg["weight_decay"])
    bs = 16
    for _ in range(hcfg["epochs"]):
        model.train()
        perm = np.random.permutation(sel)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            logits = model(Xn[b].to(device), Mt[b].to(device))
            lg = logits.reshape(-1, logits.shape[-1])
            q, w = Qt[b].to(device).reshape(-1, Qt.shape[-1]), Wt[b].to(device).reshape(-1)
            keep = w > 0
            loss = coral_loss(lg[keep], q[keep], w[keep]) if keep.any() else lg.sum() * 0.0
            if use_teacher:
                tm = TMt[b].to(device).reshape(-1)
                if tm.any():
                    tq = TQt[b].to(device).reshape(-1, Qt.shape[-1])
                    loss = (1 - alpha) * loss + alpha * coral_loss(lg[tm], tq[tm], None)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), scaler


@torch.no_grad()
def predict_seq(model: SeqOrdinal, scaler: dict, X: np.ndarray, M: np.ndarray, device: str = "cpu") -> np.ndarray:
    """-> probs (N, T, K); NaN where M is False."""
    Xn = torch.tensor(((X - scaler["mu"]) / scaler["sd"]) * M[..., None], dtype=torch.float32, device=device)
    Mt = torch.tensor(M, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(X), 32):
        lg = model(Xn[i:i + 32], Mt[i:i + 32])
        out.append(stage_probs(lg.reshape(-1, lg.shape[-1])).reshape(lg.shape[0], lg.shape[1], -1).cpu().numpy())
    P = np.concatenate(out)
    P[~M] = np.nan
    return P
