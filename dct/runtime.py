"""Device switch shared by every GPU-using command.

Precedence (first one that is set wins):
  1. --device cuda|mps|cpu   (command line; sets DCT_DEVICE for the process)
  2. DCT_DEVICE environment variable
  3. clay.device in config/us_benchmark.yaml
  4. auto: CUDA, then MPS, then CPU
Asking for a device that is not available is an error (never a silent fall back to CPU).
Per-device settings (precision, batch size) come from clay.profiles in the config.
"""
from __future__ import annotations

import os

DEVICES = ("cuda", "mps", "cpu")


def add_device_arg(ap) -> None:
    ap.add_argument("--device", choices=("auto",) + DEVICES, default=None,
                    help="compute device (default: DCT_DEVICE env var, then config, then auto: cuda > mps > cpu)")


def apply_device_arg(args) -> None:
    """Make --device win over the environment variable and the config."""
    if getattr(args, "device", None):
        if args.device == "auto":
            os.environ.pop("DCT_DEVICE", None)
        else:
            os.environ["DCT_DEVICE"] = args.device


def pick_device(prefer: str | None = None):
    """Resolve the device. `prefer` is the config value; DCT_DEVICE overrides it."""
    import torch

    choice = (os.environ.get("DCT_DEVICE") or prefer or "").strip().lower() or None
    if choice in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice not in DEVICES:
        raise ValueError(f"unknown device '{choice}'; use one of {DEVICES} or 'auto'")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False "
                           "(CPU-only torch build or no driver). Run: python -m dct.env_check")
    if choice == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but it is not available on this machine")
    return torch.device(choice)


def device_profile(ccfg: dict, device) -> dict:
    """{'dtype': 'float32'|'bfloat16'|'float16', 'batch_size': int} for this device type."""
    prof = dict((ccfg.get("profiles") or {}).get(device.type, {}))
    prof.setdefault("dtype", "float32")
    prof.setdefault("batch_size", ccfg.get("batch_size", 16))
    if prof["dtype"] not in ("float32", "bfloat16", "float16"):
        raise ValueError(f"clay.profiles.{device.type}.dtype must be float32, bfloat16 or float16")
    return prof
