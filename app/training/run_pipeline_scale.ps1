<#
SCALE paper-faithful pipeline (Windows, full ~10K).
Detectron2/torchvision regions -> S3D/resnet video -> mel audio -> LMDB ->
pretrain SIMCL -> extract tpiva -> Faiss -> eval.
#>
param(
    [string]$Python = "C:\Users\SenetUser\AppData\Local\Programs\Python\Python312\python.exe",
    [string]$DatasetDir = "app/datasets/downloaded_m5product_balanced",
    [string]$SplitsDir = "artifacts/downloaded_2k",
    [string]$WorkDir = "artifacts/scale_paper",
    [int]$Limit = 0,
    [switch]$PrepareOnly,
    [switch]$SkipFeatures,
    [switch]$SkipPretrain,
    [switch]$SkipEval,
    [int]$TrainEpochs = 5,
    [int]$BatchSize = 16,
    [int]$GradAccum = 16,
    [switch]$Fp16 = $true,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo

$scaleRoot = Join-Path $repo "app\SCALE"
$scaleExamples = Join-Path $scaleRoot "examples\SCALE"
$env:PYTHONPATH = "$scaleRoot;$repo\app"

if ($SmokeTest) {
    $Limit = 50
    $TrainEpochs = 1
}

Write-Host "=== Phase 1: Convert dataset to SCALE layout ==="
$convertArgs = @(
    "$scaleRoot\preprocess\convert_balanced_to_scale.py",
    "--dataset-dir", $DatasetDir,
    "--splits-dir", $SplitsDir,
    "--output-dir", $WorkDir
)
& $Python @convertArgs

if ($PrepareOnly) { exit 0 }

if (-not $SkipFeatures) {
    Write-Host "=== Phase 2a: Region features (TSV) ==="
    $regionArgs = @(
        "$scaleRoot\tools\bp_feature\extract\extract_regions_windows.py",
        "--id-label", "$WorkDir\id_label.json",
        "--path-manifest", "$WorkDir\path_manifest.json",
        "--output-tsv", "$WorkDir\tsv_features\features.tsv",
        "--backend", "auto"
    )
    if ($Limit -gt 0) { $regionArgs += @("--limit", $Limit) }
    & $Python @regionArgs

    Write-Host "=== Phase 2b: Video features + mp3 ==="
    $videoArgs = @(
        "$scaleRoot\preprocess\extract_video_audio.py",
        "--path-manifest", "$WorkDir\path_manifest.json",
        "--video-output-dir", "$WorkDir\video_feature",
        "--audio-output-dir", "$WorkDir\audios",
        "--zero-fill-missing",
        "--reextract-zero",
        "--device", "cpu"
    )
    if ($Limit -gt 0) { $videoArgs += @("--limit", $Limit) }
    & $Python @videoArgs

    Write-Host "=== Phase 2c: Audio mel features ==="
    & $Python "$scaleRoot\tools\audio_process\save_audio_feature.py" `
        --id-label "$WorkDir\id_label.json" `
        --audio-dir "$WorkDir\audios" `
        --output-dir "$WorkDir\audio_feature" `
        --zero-fill-missing

    Write-Host "=== Phase 3: Build LMDB splits ==="
    foreach ($split in @("train", "test", "gallery")) {
        & $Python "$scaleRoot\tools\bp_feature\convert\convert_from_config.py" `
            --tsv-dir "$WorkDir\tsv_features" `
            --output-lmdb "$WorkDir\lmdb_features\${split}_feature.lmdb" `
            --ids-file "$WorkDir\${split}_ids.json"
    }
}

if (-not $SkipPretrain) {
    Write-Host "=== Phase 4: Pretrain SIMCL ==="
    Push-Location $scaleExamples
    $pretrainArgs = @(
        "pretrain_task.py",
        "--from_pretrained", "bert-base-chinese",
        "--bert_model", "bert-base-chinese",
        "--config_file", "..\..\config\bert_base_6layer_6conect_capture_itp3va.json",
        "--predict_feature",
        "--learning_rate", "1e-4",
        "--video_feature_dir", (Resolve-Path "$repo\$WorkDir\video_feature"),
        "--audio_feature_dir", (Resolve-Path "$repo\$WorkDir\audio_feature"),
        "--lmdb_file", (Resolve-Path "$repo\$WorkDir\lmdb_features\train_feature.lmdb"),
        "--caption_path", (Resolve-Path "$repo\$WorkDir\id_label.json"),
        "--output_dir", (Resolve-Path "$repo\$WorkDir\checkpoints"),
        "--save_name", "scale_paper_simcl",
        "--train_batch_size", $BatchSize,
        "--gradient_accumulation_steps", $GradAccum,
        "--num_train_epochs", $TrainEpochs,
        "--max_seq_length", "36",
        "--video_len", "12",
        "--pv_seq_len", "64",
        "--audio_len", "12",
        "--num_workers", "0",
        "--MLM", "--MRM", "--MEM", "--MFM", "--MAM", "--CLR"
    )
    if ($Fp16) { $pretrainArgs += "--fp16" }
    & $Python @pretrainArgs 2>&1 | Tee-Object -FilePath "$repo\$WorkDir\pretrain.log"
    Pop-Location
}

Write-Host "=== Phase 5: Extract tpiva features ==="
$ckptDir = Join-Path "$repo\$WorkDir\checkpoints" "scale_paper_simcl"
$ckpt = Get-ChildItem $ckptDir -Filter "pytorch_model_*.bin" | Sort-Object { [int]($_.BaseName -replace 'pytorch_model_','') } -Descending | Select-Object -First 1
if (-not $ckpt) { throw "No pytorch_model_*.bin found under $ckptDir" }

Push-Location $scaleExamples
foreach ($split in @("test", "gallery")) {
    $featDir = "$repo\$WorkDir\features\$split"
    New-Item -ItemType Directory -Force -Path $featDir | Out-Null
    & $Python eval\extract_features.py `
        --from_pretrained $ckpt.FullName `
        --bert_model bert-base-chinese `
        --config_file ..\..\config\bert_base_6layer_6conect_capture_itp3va.json `
        --predict_feature `
        --lmdb_file "$repo\$WorkDir\lmdb_features\${split}_feature.lmdb" `
        --caption_path "$repo\$WorkDir\id_label.json" `
        --video_feature_dir "$repo\$WorkDir\video_feature" `
        --audio_feature_dir "$repo\$WorkDir\audio_feature" `
        --feature_dir $featDir `
        --train_batch_size 8 `
        --max_seq_length 36 `
        --video_len 12 `
        --pv_seq_len 64 `
        --audio_len 12 `
        --num_workers 0 `
        --split $split
}
Pop-Location

Write-Host "=== Phase 5b: Retrieval eval ==="
if (-not $SkipEval) {
    & $Python "$scaleRoot\preprocess\run_retrieval_eval.py" `
        --query-features "$WorkDir\features\test\tpiva_feature_np.npy" `
        --query-ids "$WorkDir\features\test\id.npy" `
        --gallery-features "$WorkDir\features\gallery\tpiva_feature_np.npy" `
        --gallery-ids "$WorkDir\features\gallery\id.npy" `
        --id-label "$WorkDir\id_label.json" `
        --output "$WorkDir\evaluation.json"
}

Write-Host "=== Phase 6: Faiss index ==="
& $Python app\indexing\build_index.py `
    --embeddings "$WorkDir\features\gallery\tpiva_feature_np.npy" `
    --ids "$WorkDir\features\gallery\id.npy" `
    --output-dir "$WorkDir\index_hnsw" `
    --index-type hnsw

Write-Host "Pipeline complete."
Write-Host "  Work dir: $WorkDir"
Write-Host "  Eval: $WorkDir\evaluation.json"
Write-Host "  API: `$env:SCALE_BACKEND='paper'; `$env:SCALE_WORK_DIR='$WorkDir'; uvicorn api.main:app --app-dir app --reload --port 8000"
