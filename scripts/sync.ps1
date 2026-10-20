# Pull the latest code and check the machine still works.   powershell -ExecutionPolicy Bypass -File scripts/sync.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
git pull --rebase --autostash
micromamba run -n dct-ml python -m dct.env_check
