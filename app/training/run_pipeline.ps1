<#
SCALE + Faiss retrieval pipeline.
Five modalities (image, text, table/pv, video, audio) with zero imputation for missing inputs.
#>
param(
    [string]$Python = "python",
    [string]$DatasetDir = "app/datasets/downloaded_2k",
    [string]$WorkDir = "artifacts/downloaded_2k",
    [string]$Model = "google/siglip-base-patch16-224",
    [switch]$SkipCaptioning,
    [switch]$PrepareOnly,
    [switch]$SkipTraining,
    [switch]$SkipEvaluation,
    [int]$TrainEpochs = 3,
    [int]$BatchSize = 8,
    [string]$Device = ""
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo
$env:PYTHONPATH = (Join-Path $repo "app")

if (-not $Device) {
    $Device = & $Python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
}
$deviceArg = @()
if ($Device -and $Device -ne "cpu") { $deviceArg = @("--device", $Device) }

& $Python -c "import torch; print('device:', 'cuda' if torch.cuda.is_available() else 'cpu'); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

& $Python app/preprocess/prepare_subset.py `
    --dataset-dir $DatasetDir `
    --output-dir $WorkDir

if (-not $SkipCaptioning) {
    & $Python app/preprocess/generate_descriptions.py `
        --records "$WorkDir/records.json"
}

if ($PrepareOnly) {
    exit 0
}

if (-not $SkipTraining) {
    & $Python app/scale_runtime/train_fusion.py `
        --records "$WorkDir/records.json" `
        --split "$WorkDir/train.json" `
        --output "$WorkDir/fusion_weights.json" `
        --epochs $TrainEpochs `
        --batch-size $BatchSize `
        @deviceArg
}

& $Python app/encoding/extract_scale_embeddings.py `
    --records "$WorkDir/records.json" `
    --splits-dir $WorkDir `
    --output-dir "$WorkDir/embeddings" `
    --model $Model `
    --fusion-weights "$WorkDir/fusion_weights.json" `
    --splits gallery test `
    @deviceArg

& $Python app/indexing/build_index.py `
    --embeddings "$WorkDir/embeddings/gallery/embedding.npy" `
    --ids "$WorkDir/embeddings/gallery/id.npy" `
    --output-dir "$WorkDir/index_hnsw" `
    --index-type hnsw

if (-not $SkipEvaluation) {
    & $Python app/evaluation/run_retrieval_eval.py `
        --query-embeddings "$WorkDir/embeddings/test/embedding.npy" `
        --query-ids "$WorkDir/embeddings/test/id.npy" `
        --gallery-embeddings "$WorkDir/embeddings/gallery/embedding.npy" `
        --gallery-ids "$WorkDir/embeddings/gallery/id.npy" `
        --records "$WorkDir/records.json" `
        --index-dir "$WorkDir/index_hnsw" `
        --output "$WorkDir/evaluation.json"
}

Write-Host "Pipeline complete. Start API with:"
Write-Host "  `$env:PYTHONPATH='app'; `$env:SCALE_WORK_DIR='$WorkDir'; uvicorn api.main:app --app-dir app --reload --port 8000"
