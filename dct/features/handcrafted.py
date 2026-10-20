"""Hand-made Sentinel-2 / radar features for the explainable baseline (model A) and the evidence pack.

Every feature is computed twice, INSIDE the campus footprint and in a surrounding RING, and the
difference (inside - ring) is kept too: a data center under construction changes relative to its
surroundings, while seasons and haze change both together.

Sentinel-2 band order here = Clay's: blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22
(Earth Search reflectance DN; no baseline offset correction is applied - live pixels show the archive is already
corrected; indices are ratios so the scale cancels anyway).
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.ops import transform as shp_transform

S2_ORDER = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
B = {name: i for i, name in enumerate(S2_ORDER)}
EPS = 1e-6


def pixel_masks(footprint_4326, transform, crs, size_px: int, ring_m: float) -> tuple[np.ndarray, np.ndarray]:
    """(inside, ring) boolean pixel masks (size_px, size_px) for a chip."""
    fp = shp_transform(Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform, footprint_4326)
    inside = geometry_mask([fp], out_shape=(size_px, size_px), transform=transform, invert=True)
    outer = geometry_mask([fp.buffer(ring_m)], out_shape=(size_px, size_px), transform=transform, invert=True)
    ring = outer & ~inside
    return inside, ring


def s2_layers(chip: np.ndarray) -> dict[str, np.ndarray]:
    """chip (10, H, W) float reflectance -> named 2-D layers."""
    f = lambda n: chip[B[n]].astype("float32")
    red, nir, sw1, sw2, blue, green = f("red"), f("nir"), f("swir16"), f("swir22"), f("blue"), f("green")
    return {
        "ndvi": (nir - red) / (nir + red + EPS),                         # vegetation
        "ndbi": (sw1 - nir) / (sw1 + nir + EPS),                         # built-up
        "bsi": ((sw1 + red) - (nir + blue)) / ((sw1 + red) + (nir + blue) + EPS),   # bare soil
        "mndwi": (green - sw1) / (green + sw1 + EPS),                    # water / wet ground
        "bright": (red + green + blue) / 3.0,                            # visible brightness
        "swir2": sw2,                                                    # roofs and bare ground
    }


def s2_stats(chip: np.ndarray, inside: np.ndarray, ring: np.ndarray, bad: np.ndarray | None = None) -> dict[str, float]:
    """Feature dict: <layer>_in, <layer>_ring, <layer>_diff for each layer, plus bright_std_in (texture).
    bad = optional (H, W) bool mask of cloud / shadow / cirrus / no-data pixels (from the SCL) to leave out."""
    out: dict[str, float] = {}
    layers = s2_layers(chip)
    valid = chip.sum(0) > 0
    if bad is not None:
        valid = valid & ~bad
    mi, mr = inside & valid, ring & valid
    for name, arr in layers.items():
        vi = float(arr[mi].mean()) if mi.any() else np.nan
        vr = float(arr[mr].mean()) if mr.any() else np.nan
        out[f"{name}_in"], out[f"{name}_ring"], out[f"{name}_diff"] = vi, vr, vi - vr
    out["bright_std_in"] = float(layers["bright"][mi].std()) if mi.sum() > 1 else np.nan
    out["n_in_px"], out["n_ring_px"] = int(mi.sum()), int(mr.sum())
    return out


FEATURE_COLUMNS = [f"{n}_{s}" for n in ("ndvi", "ndbi", "bsi", "mndwi", "bright", "swir2") for s in ("in", "ring", "diff")] + ["bright_std_in"]


def db(x: np.ndarray) -> np.ndarray:
    """Linear power -> dB (OPERA RTC-S1 gamma0 is stored as linear power)."""
    return 10.0 * np.log10(np.clip(x, 1e-6, None))
