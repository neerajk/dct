"""Check this machine's GPU stack and (optionally) tune the Clay encoder for it.

Run (ML env):
  python -m dct.env_check                          # torch / CUDA / driver / TerraTorch / GPU op test
  python -m dct.env_check --clay                   # + load Clay v1.5, fp32 vs the device's reduced-precision profile
  python -m dct.env_check --clay --probe-batch     # + throughput and peak memory per batch size (picks the best)
  python -m dct.env_check --clay --device cpu      # force a device

What the checks tell you
  * 'torch.version.cuda' is None  -> a CPU-only torch was installed; reinstall from the cu128 index (README).
  * 'no kernel image' on the GPU op test -> the torch build does not include your GPU architecture (the RTX 50 series
    is Blackwell, compute capability 12.0, which needs a CUDA 12.8+ build of torch).
  * the parity numbers compare Clay embeddings computed in float32 and in the profile precision (bfloat16 on CUDA).
    Cosine similarity of the CLS vector should be above 0.999; if not, set clay.profiles.<device>.dtype to float32.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np

import dct  # noqa: F401  (sets the env vars before torch is imported)
from dct.runtime import add_device_arg, apply_device_arg


def _gpu_report(torch) -> bool:
    ok = True
    print(f"python {sys.version.split()[0]} on {platform.platform()}")
    print(f"torch {torch.__version__}; built for CUDA {torch.version.cuda}; cuDNN {torch.backends.cudnn.version()}")
    try:
        import terratorch
        print(f"terratorch {terratorch.__version__}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"terratorch: NOT importable ({exc})")
        ok = False
    if torch.version.cuda is None and not torch.backends.mps.is_available():
        print("WARNING: this torch build has no CUDA support (CPU-only wheel). See README, 'Windows + NVIDIA'.")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU 0: {p.name}; compute capability {p.major}.{p.minor}; {p.total_memory / 2**30:.1f} GiB VRAM; "
              f"bf16 supported: {torch.cuda.is_bf16_supported()}")
        try:
            a = torch.randn(2048, 2048, device="cuda")
            b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
            torch.cuda.synchronize()
            print(f"GPU op test: fp32 sum {float((a @ a).sum()):.3e}, bf16 sum {float((b @ b).float().sum()):.3e}  OK")
        except Exception as exc:                               # noqa: BLE001
            print(f"GPU op test FAILED: {exc}")
            ok = False
    elif torch.backends.mps.is_available():
        print("Apple MPS available")
    else:
        print("no GPU visible to torch (cuda.is_available() is False): the pipeline would run on CPU")
    return ok


def _fake_chips(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    chips = rng.uniform(200, 3000, size=(n, 10, 256, 256)).astype("float32")      # S2-like reflectance units
    times = [[0.1, 0.9, 0.0, 1.0]] * n
    latlons = [[0.62, 0.78, -0.99, 0.17]] * n
    return chips, times, latlons


def _clay_checks(cfg: dict, probe: bool) -> bool:
    import torch
    from dct.clay.encoder import ClayEncoder

    t0 = time.time()
    enc = ClayEncoder(cfg)
    print(f"Clay v1.5 loaded on {enc.dev} in {time.time() - t0:.0f}s; profile dtype {enc.amp_dtype or 'float32'}, "
          f"batch {enc.batch_size}")
    plat = "sentinel-2-l2a"
    chips, times, latlons = _fake_chips(4)
    profile_dtype = enc.amp_dtype
    enc.amp_dtype = None
    ref = enc.tokens(chips, plat, times, latlons)
    enc.amp_dtype = profile_dtype
    if profile_dtype is not None:
        got = enc.tokens(chips, plat, times, latlons)
        cos_cls = torch.nn.functional.cosine_similarity(ref[:, 0], got[:, 0], dim=-1).min().item()
        cos_tok = torch.nn.functional.cosine_similarity(ref[:, 1:].reshape(-1, ref.shape[-1]),
                                                        got[:, 1:].reshape(-1, ref.shape[-1]), dim=-1).min().item()
        print(f"parity fp32 vs {profile_dtype}: min cosine CLS {cos_cls:.5f}, patch tokens {cos_tok:.5f} "
              f"({'OK' if cos_cls > 0.999 else 'LOW: use dtype float32 for this device'})")
    else:
        print("profile is float32: no parity check needed")
    print(f"embedding shape {tuple(ref.shape)} (expect (4, 1025, 1024)); finite: {bool(torch.isfinite(ref).all())}")
    if probe:
        print("batch  chips/s  peak GPU MiB")
        best = None
        for bs in (4, 8, 16, 32, 64, 96, 128):
            chips, times, latlons = _fake_chips(bs)
            try:
                if enc.dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                enc.tokens(chips, plat, times, latlons)                     # warm-up
                t0 = time.time()
                for _ in range(2):
                    enc.tokens(chips, plat, times, latlons)
                if enc.dev.type == "cuda":
                    torch.cuda.synchronize()
                rate = 2 * bs / (time.time() - t0)
                mem = torch.cuda.max_memory_allocated() / 2**20 if enc.dev.type == "cuda" else float("nan")
                print(f"{bs:5d}  {rate:7.1f}  {mem:10.0f}")
                if best is None or rate > best[1] * 1.05:
                    best = (bs, rate)
            except RuntimeError as exc:
                print(f"{bs:5d}  failed ({str(exc).splitlines()[0][:70]})")
                break
        if best:
            print(f"-> set clay.profiles.{enc.dev.type}.batch_size to about {best[0]} (best gain above 5% per step)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clay", action="store_true", help="load Clay v1.5 and check precision parity")
    ap.add_argument("--probe-batch", action="store_true", help="with --clay: measure throughput per batch size")
    add_device_arg(ap)
    args = ap.parse_args()
    apply_device_arg(args)
    import torch
    ok = _gpu_report(torch)
    if args.clay:
        from dct.common import load_us_config
        ok = _clay_checks(load_us_config(), args.probe_batch) and ok
    print("\nRESULT:", "OK" if ok else "PROBLEMS FOUND (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
