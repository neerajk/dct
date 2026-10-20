# One-time setup on Windows with an NVIDIA GPU (RTX 50 series needs CUDA 12.8+ torch wheels).
# Run from the repo root in PowerShell:   powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
# Needs: micromamba on PATH, a recent NVIDIA driver (nvidia-smi works), git.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

# Windows Python defaults to cp1252 for text files; make UTF-8 the default for this user from now on.
[Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
$env:PYTHONUTF8 = "1"

nvidia-smi | Select-Object -First 4

micromamba create -y -f environment-ml-cuda.yml
micromamba run -n dct-ml pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
micromamba run -n dct-ml pip install "terratorch>=1.2.12"

# TerraTorch may have replaced torch with a CPU wheel; if so put the CUDA build back.
$cuda = micromamba run -n dct-ml python -c "import torch; print(torch.version.cuda)"
if ($cuda -match "None") {
    Write-Host "CPU-only torch detected after installing terratorch; reinstalling the CUDA build"
    micromamba run -n dct-ml pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu128
}

micromamba run -n dct-ml python scripts/get_weights.py
micromamba run -n dct-ml python -m dct.env_check --clay --probe-batch
Write-Host "`nDone. Activate with:  micromamba activate dct-ml"
