"""Download the Clay v1.5 checkpoint (5.16 GB, Apache-2.0) into models/clay/ and verify it.

Run from the repo root (needs only `requests`):  python scripts/get_weights.py
  * resumable: an interrupted download continues from models/clay/clay-v1.5.ckpt.part
  * verified against the SHA-256 and size that Hugging Face publishes for this file (checked 2026-10-20)
Source: https://huggingface.co/made-with-clay/Clay (v1.5/clay-v1.5.ckpt)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

URL = "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"
SHA256 = "21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0"
SIZE = 5158764629
DEST = Path(__file__).resolve().parents[1] / "models" / "clay" / "clay-v1.5.ckpt"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def verified(path: Path) -> bool:
    return path.exists() and path.stat().st_size == SIZE and sha256_of(path) == SHA256


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"{DEST} exists; verifying SHA-256 (about a minute) ...")
        if verified(DEST):
            print("OK: checkpoint is complete and matches the published hash.")
            return 0
        print("existing file does not match; moving it aside and downloading again")
        DEST.replace(DEST.with_suffix(".ckpt.bad"))

    part = DEST.with_name(DEST.name + ".part")
    for attempt in range(1, 6):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:                       # already have every byte
                    break
                r.raise_for_status()
                if have and r.status_code != 206:              # server ignored the Range header: start over
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    done = have
                    for block in r.iter_content(8 * 2**20):
                        f.write(block)
                        done += len(block)
                        print(f"\r{done / 2**30:5.2f} / {SIZE / 2**30:.2f} GiB", end="", flush=True)
            print()
            break
        except (requests.RequestException, OSError) as exc:
            print(f"\nattempt {attempt} failed ({exc}); retrying (the download resumes)")
    else:
        print("download failed after 5 attempts; run the script again to resume")
        return 1

    if part.stat().st_size != SIZE:
        print(f"size mismatch: {part.stat().st_size} bytes, expected {SIZE}; run again to resume")
        return 1
    print("verifying SHA-256 ...")
    if sha256_of(part) != SHA256:
        part.unlink()
        print("hash mismatch: partial file deleted; run the script again")
        return 1
    part.replace(DEST)
    print(f"OK: {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
