"""Pool Clay patch tokens inside a campus footprint and in a surrounding ring.

A 256 px chip gives a 32 x 32 grid of 8 px patches (80 m for Sentinel-2, 8 m for NAIP at 1 m).
'inside' = patches whose centre is within the footprint (grown by half a patch, so patches that
only overlap the outline still count); 'ring' = patches within ring_m of the footprint but not
inside. If nothing falls inside, the patch nearest the footprint centroid is used. If the whole chip is inside
(a very large campus) there is no ring: the ring then copies 'inside' so the feature inside - ring is 0, not the raw embedding.
Stored per chip: cls (1024), inside (1024), ring (1024) - the ring tells the model what is NOT the
data center (surrounding fields, roads), which is what makes change detectable.
"""
from __future__ import annotations

import numpy as np
import shapely
import torch
from pyproj import Transformer
from shapely.ops import transform as shp_transform


def patch_masks(footprint_4326, transform, crs, size_px: int, patch_px: int, gsd_m: float, ring_m: float):
    """Boolean (grid, grid) masks (inside, ring) for a chip with the given affine transform and CRS."""
    grid = size_px // patch_px
    idx = (np.arange(grid) + 0.5) * patch_px
    xs = transform.c + idx * transform.a
    ys = transform.f + idx * transform.e
    X, Y = np.meshgrid(xs, ys)
    fp = shp_transform(Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform, footprint_4326)
    half = patch_px * gsd_m / 2.0
    inside = shapely.contains_xy(fp.buffer(half), X, Y)
    if not inside.any():
        c = fp.centroid
        d = (X - c.x) ** 2 + (Y - c.y) ** 2
        inside = d == d.min()
    ring = shapely.contains_xy(fp.buffer(ring_m + half), X, Y) & ~inside
    if not ring.any():
        ring = inside.copy()      # chip lies wholly inside the footprint: no outside context, so inside - ring = 0 (neutral)
    return inside, ring


def pool_tokens(tokens: torch.Tensor, inside: torch.Tensor, ring: torch.Tensor) -> dict[str, np.ndarray]:
    """tokens (B, 1+L, D); inside/ring (B, g, g) bool with g*g == L. Returns float32 arrays (B, D)."""
    cls = tokens[:, 0]
    patches = tokens[:, 1:]
    out = {"cls": cls.numpy()}
    for name, m in (("inside", inside), ("ring", ring)):
        w = m.reshape(m.shape[0], -1).float()
        out[name] = ((patches * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1.0)).numpy()
    return out


def embed_chips(enc, chips: np.ndarray, platform: str, times: list, latlons: list,
                inside: np.ndarray, ring: np.ndarray, batch_size: int = 16) -> dict[str, np.ndarray]:
    """chips (N, C, H, W) raw units; times/latlons lists of 4-vectors; inside/ring (N, g, g) bool."""
    parts = {"cls": [], "inside": [], "ring": []}
    for i in range(0, len(chips), batch_size):
        sl = slice(i, i + batch_size)
        tok = enc.tokens(chips[sl], platform, times[sl], latlons[sl])
        pooled = pool_tokens(tok, torch.from_numpy(inside[sl]), torch.from_numpy(ring[sl]))
        for k in parts:
            parts[k].append(pooled[k])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}
