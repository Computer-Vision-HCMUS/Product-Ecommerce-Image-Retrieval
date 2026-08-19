$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = if ($env:SCALE_PYTHON) { $env:SCALE_PYTHON } else { "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" }
$env:PYTHONPATH = Join-Path $repo "app"
$env:SCALE_BACKEND = "paper"
$env:SCALE_WORK_DIR = Join-Path $repo "artifacts\scale_paper"
if (-not $env:SCALE_PIPELINE) { $env:SCALE_PIPELINE = "baseline" }
Set-Location $repo
& $py -m uvicorn api.main:app --app-dir app --host 127.0.0.1 --port 8000 --reload
