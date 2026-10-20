"""Clay v1.5 encoder, run through TerraTorch's vendored Encoder (terratorch >= 1.2.12).

Why not Clay1_5ModelFactory: its backbone wrapper (terratorch/models/clay1_5_model_factory.py,
ClayMAEBackbone.forward) feeds ZERO time and lat/lon. Here we build the same Encoder with the same
checkpoint loading, but call to_patch_embed / add_encodings / transformer ourselves so real
acquisition time and location are used, and no patch is masked.

Clay v1.5 input contract (Clay spec + terratorch source):
  pixels  (B, C, H, W) normalised with the per-sensor mean/std in config/clay/metadata.yaml
  waves   (C,)  band wavelengths in micrometres, from the same file (SAR 'wavelengths' are placeholders)
  gsd     scalar, metres per pixel
  time    (B, 4) = [sin week, cos week, sin hour, cos hour]
  latlon  (B, 4) = [sin lat, cos lat, sin lon, cos lon]  (radians)
The time/latlon encodings match Clay's stacchip prechip.py (normalize_timestamp: ISO week * 2pi/52 and hour * 2pi/24;
normalize_latlon: radians), as checked against the upstream source in the code review of 2026-10-20.

Output tokens: (B, 1 + (H/8)*(W/8), 1024): CLS first, then the patch grid (32 x 32 for 256 px chips).
"""
from __future__ import annotations

import contextlib
import math
import os
from datetime import date, datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")          # OpenMP clash on this machine
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")      # let unsupported MPS ops fall back to CPU

import numpy as np
import torch
import yaml
from einops import repeat
from torch import nn

from dct.common import REPO_ROOT
from dct.runtime import device_profile, pick_device   # noqa: F401  (pick_device re-exported)




def encode_time(d: date | datetime, hour: int = 12) -> list[float]:
    week = d.isocalendar()[1] * 2 * math.pi / 52
    h = hour * 2 * math.pi / 24
    return [math.sin(week), math.cos(week), math.sin(h), math.cos(h)]


def encode_latlon(lat: float, lon: float) -> list[float]:
    la, lo = math.radians(lat), math.radians(lon)
    return [math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)]


class ClayEncoder(nn.Module):
    def __init__(self, cfg: dict, device: str | None = None):
        super().__init__()
        from terratorch.models.backbones.clay_v15.model import Encoder   # needs terratorch >= 1.2.12

        ccfg = cfg["us"]["clay"]
        e = ccfg["encoder"]
        self.encoder = Encoder(mask_ratio=0.0, patch_size=e["patch_size"], shuffle=False, dim=e["dim"],
                               depth=e["depth"], heads=e["heads"], dim_head=e["dim_head"], mlp_ratio=e["mlp_ratio"])
        self._load_weights(REPO_ROOT / ccfg["checkpoint"], ccfg["weights_prefix"])
        with open(REPO_ROOT / ccfg["metadata"], encoding="utf-8") as f:
            self.meta = yaml.safe_load(f)
        self.dim = e["dim"]
        self.patch_size = e["patch_size"]
        self.dev = pick_device(device or ccfg.get("device"))
        prof = device_profile(ccfg, self.dev)
        self.batch_size = int(prof["batch_size"])
        # reduced precision only on the accelerator profile; None = plain float32 (set to None to force fp32)
        self.amp_dtype = None if prof["dtype"] == "float32" else getattr(torch, prof["dtype"])
        self.to(self.dev).eval()

    def _load_weights(self, path: Path, prefix: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Clay checkpoint not found: {path}")
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except Exception:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        enc = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not enc:
            raise RuntimeError(f"no '{prefix}*' keys in {path}; first keys: {list(state)[:8]}")
        missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
        del ckpt, state
        if missing:
            raise RuntimeError(f"Clay encoder keys missing after load (wrong dim_head/mlp_ratio?): {missing[:6]}")
        if unexpected:
            print(f"note: {len(unexpected)} unexpected encoder keys ignored, e.g. {unexpected[:3]}")

    # ----- per-sensor normalisation -------------------------------------------------------------
    def band_info(self, platform: str) -> dict:
        m = self.meta[platform]
        order = m["band_order"]
        return {"order": order, "gsd": float(m["gsd"]),
                "mean": np.array([m["bands"]["mean"][b] for b in order], dtype="float32"),
                "std": np.array([m["bands"]["std"][b] for b in order], dtype="float32"),
                "waves": np.array([m["bands"]["wavelength"][b] for b in order], dtype="float32")}

    def normalise(self, arr: np.ndarray, platform: str) -> np.ndarray:
        """arr (B, C, H, W) in the sensor's raw units and metadata band order -> normalised float32."""
        info = self.band_info(platform)
        if arr.shape[1] != len(info["order"]):
            raise ValueError(f"{platform} expects {len(info['order'])} bands {info['order']}, got {arr.shape[1]}")
        return (arr.astype("float32") - info["mean"][None, :, None, None]) / info["std"][None, :, None, None]

    # ----- forward ----------------------------------------------------------------------------------
    @torch.no_grad()
    def tokens(self, pixels: np.ndarray, platform: str, times: list[list[float]], latlons: list[list[float]]) -> torch.Tensor:
        """pixels: raw (B, C, H, W). Returns tokens (B, 1 + L, D) on CPU float32."""
        info = self.band_info(platform)
        x = torch.from_numpy(self.normalise(pixels, platform)).to(self.dev)
        waves = torch.from_numpy(info["waves"]).to(self.dev)
        gsd = torch.tensor(info["gsd"], device=self.dev)
        time = torch.tensor(times, dtype=torch.float32, device=self.dev)
        latlon = torch.tensor(latlons, dtype=torch.float32, device=self.dev)
        B = x.shape[0]
        amp = (torch.autocast(device_type=self.dev.type, dtype=self.amp_dtype) if self.amp_dtype is not None
               else contextlib.nullcontext())
        with amp:
            patches, _ = self.encoder.to_patch_embed(x, waves)
            patches = self.encoder.add_encodings(patches, time, latlon, gsd)
            cls = repeat(self.encoder.cls_token, "1 1 D -> B 1 D", B=B)
            out = self.encoder.transformer(torch.cat((cls, patches), dim=1))
        return out.float().cpu()
