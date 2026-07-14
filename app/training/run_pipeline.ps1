<#
Runs the SigLIP + BLIP + Faiss pipeline from an Anaconda Prompt/initialized
PowerShell. SigLIP is pretrained; no SCALE pretraining or LMDB is required.
#>
param(
    [string]$Python = "python",
    [string]$DatasetDir = "app/datasets/downloaded_2k",
    [string]$WorkDir = "artifacts/downloaded_2k",
    [string]$Model = "google/siglip-base-patch16-224",
    [double]$ImageWeight = 0.7,
    [switch]$SkipCaptioning,
    [switch]$PrepareOnly,
    [switch]$SkipEvaluation
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo

& $Python -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Anaconda base with Python 3.12.'"
& $Python -c "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA is required.'; print(torch.cuda.get_device_name(0))"

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

& $Python app/encoding/extract_embeddings.py `
    --records "$WorkDir/records.json" `
    --splits-dir $WorkDir `
    --output-dir "$WorkDir/embeddings" `
    --model $Model `
    --image-weight $ImageWeight `
    --splits gallery query

& $Python app/indexing/build_index.py `
    --embeddings "$WorkDir/embeddings/gallery/embedding.npy" `
    --ids "$WorkDir/embeddings/gallery/id.npy" `
    --output-dir "$WorkDir/index_hnsw" `
    --index-type hnsw

if (-not $SkipEvaluation) {
    & $Python app/evaluation/run_retrieval_eval.py `
        --query-embeddings "$WorkDir/embeddings/query/embedding.npy" `
        --query-ids "$WorkDir/embeddings/query/id.npy" `
        --gallery-embeddings "$WorkDir/embeddings/gallery/embedding.npy" `
        --gallery-ids "$WorkDir/embeddings/gallery/id.npy" `
        --records "$WorkDir/records.json" `
        --index-dir "$WorkDir/index_hnsw" `
        --output "$WorkDir/evaluation.json"
}
