# Start SCALE paper pretrain — single instance, live log, GPU training.
param(
    [int]$TrainEpochs = 1
)
$ErrorActionPreference = "Continue"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repo

$py = "C:\Users\SenetUser\AppData\Local\Programs\Python\Python312\python.exe"
$work = Join-Path $repo "artifacts\scale_paper"
$log = Join-Path $work "pretrain.log"
$examples = Join-Path $repo "app\SCALE\examples\SCALE"
$env:PYTHONPATH = "$repo\app\SCALE;$repo\app"
$env:PYTHONUNBUFFERED = "1"

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'pretrain_task\.py' }
if ($running) {
    Write-Host "Pretrain already running (PID $($running.ProcessId)). Stop it first."
    exit 1
}

"" | Set-Content -Path $log -Encoding utf8
Write-Host "Epochs: $TrainEpochs | Log: $log"
Write-Host "Tail log: Get-Content '$log' -Wait -Tail 15"

Push-Location $examples
& $py -u pretrain_task.py `
    --from_pretrained bert-base-chinese `
    --bert_model bert-base-chinese `
    --config_file "..\..\config\bert_base_6layer_6conect_capture_itp3va.json" `
    --predict_feature `
    --learning_rate 1e-4 `
    --video_feature_dir "$work\video_feature" `
    --audio_feature_dir "$work\audio_feature" `
    --lmdb_file "$work\lmdb_features\train_feature.lmdb" `
    --caption_path "$work\id_label.json" `
    --output_dir "$work\checkpoints" `
    --save_name scale_paper_simcl `
    --train_batch_size 16 `
    --gradient_accumulation_steps 8 `
    --num_train_epochs $TrainEpochs `
    --max_seq_length 36 `
    --video_len 12 `
    --pv_seq_len 64 `
    --audio_len 12 `
    --num_workers 0 `
    --MLM --MRM --MEM --MFM --MAM --CLR `
    2>&1 | ForEach-Object {
        $_
        Add-Content -Path $log -Value $_ -Encoding utf8
    }
Pop-Location
