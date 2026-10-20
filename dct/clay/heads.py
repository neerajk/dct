"""Ordinal (CORAL-style) heads on pooled Clay embeddings, shared by the NAIP teacher and the S2 models.

CORAL (Cao et al. 2020): K ordered stages -> K-1 binary questions 'is the stage > k?' that share one
weight vector and differ only in their bias. Here the biases are forced to decrease with k, so
P(stage > k) is monotone and every stage probability is >= 0.
Targets are SOFT stage distributions q (N, K): a hard label is one-hot, a monthly label near a
transition is spread over two stages, a teacher output is its probability vector. The loss is the
sample-weighted binary cross-entropy against t_k = sum_{j>k} q_j.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def pooled_features(a: dict[str, np.ndarray]) -> np.ndarray:
    """Per-chip feature vector: inside-footprint embedding and (inside - surrounding ring)."""
    return np.concatenate([a["inside"], a["inside"] - a["ring"]], axis=1)


class OrdinalMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, n_stages: int = 5, dropout: float = 0.2):
        super().__init__()
        self.n_stages = n_stages
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.score = nn.Linear(hidden, 1, bias=False)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.raw = nn.Parameter(torch.zeros(n_stages - 1))

    def thresholds(self) -> torch.Tensor:
        return self.b0 - torch.cumsum(F.softplus(self.raw), dim=0)       # strictly decreasing in k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(self.body(x)) + self.thresholds()                # (B, K-1) logits for 'stage > k'


def stage_probs(logits: torch.Tensor) -> torch.Tensor:
    """(B, K-1) logits -> (B, K) stage probabilities."""
    p = torch.sigmoid(logits)
    first = 1.0 - p[:, :1]
    mid = p[:, :-1] - p[:, 1:]
    last = p[:, -1:]
    return torch.cat([first, mid, last], dim=1).clamp(min=0.0)


def soft_to_thresholds(q: torch.Tensor) -> torch.Tensor:
    """(B, K) stage distribution -> (B, K-1) targets P(stage > k)."""
    return 1.0 - torch.cumsum(q, dim=1)[:, :-1]


def coral_loss(logits: torch.Tensor, q: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    t = soft_to_thresholds(q).clamp(0.0, 1.0)
    per = F.binary_cross_entropy_with_logits(logits, t, reduction="none").sum(1)
    if w is not None:
        per = per * w
        return per.sum() / w.sum().clamp(min=1e-8)
    return per.mean()


def fit_head(X: np.ndarray, Q: np.ndarray, W: np.ndarray | None, hcfg: dict, n_stages: int = 5,
             seed: int = 0, device: str = "cpu") -> tuple[OrdinalMLP, dict]:
    """Train one head on standardised-inside features. Returns (model, scaler)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    scaler = {"mu": mu, "sd": sd}
    xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=device)
    qt = torch.tensor(Q, dtype=torch.float32, device=device)
    wt = None if W is None else torch.tensor(W, dtype=torch.float32, device=device)
    model = OrdinalMLP(X.shape[1], hcfg["hidden"], n_stages, hcfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hcfg["lr"], weight_decay=hcfg["weight_decay"])
    bs = min(256, len(xt))
    for _ in range(hcfg["epochs"]):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]
            loss = coral_loss(model(xt[b]), qt[b], None if wt is None else wt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), scaler


@torch.no_grad()
def predict_probs(model: OrdinalMLP, scaler: dict, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    x = torch.tensor((X - scaler["mu"]) / scaler["sd"], dtype=torch.float32, device=device)
    return stage_probs(model(x)).cpu().numpy()
