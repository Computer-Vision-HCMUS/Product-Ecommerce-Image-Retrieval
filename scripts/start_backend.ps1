$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = "C:\Users\SenetUser\AppData\Local\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = Join-Path $repo "app"
$env:SCALE_WORK_DIR = Join-Path $repo "artifacts\smoke_demo"
$env:SCALE_FUSION_WEIGHTS = Join-Path $repo "artifacts\smoke_demo\fusion_weights.json"
Set-Location $repo
& $py -m uvicorn api.main:app --app-dir app --host 127.0.0.1 --port 8000 --reload
