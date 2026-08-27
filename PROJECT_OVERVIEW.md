# SCALE - Multi-modal Product Retrieval (M5Product)

Pipeline end-to-end tren Windows: **download dataset -> split -> extract features -> pretrain SCALE (SIMCL) -> evaluate (paper metric) -> Faiss API**.

Co hai pipeline trong repo:

| Pipeline | Script | Mo ta |
|----------|--------|-------|
| **SCALE paper (khuyen nghi)** | `app/training/run_pipeline_scale.ps1` | 5 modality, BERT + JCT, loss SIMCL, metric `evaluate_unit_v2` |
| SigLIP simplified | `app/training/run_pipeline.ps1` | Fusion SigLIP nhe, khong phai paper SCALE |

Tai lieu nay mo ta **SCALE paper pipeline**.

---

## Yeu cau

- **OS:** Windows 10/11
- **Python:** 3.12
- **GPU:** NVIDIA (khuyen nghi >=6 GB VRAM)
- **CUDA PyTorch:** cai wheel phu hop tu [pytorch.org](https://pytorch.org/get-started/locally/)

```powershell
cd Y:\SCALE
python -m pip install -r app/requirements-windows.txt
```

**Tuy chon:** Detectron2 cho region feature chuan paper. Neu khong cai, script tu fallback **torchvision Faster R-CNN** (feature pad len 2048-d).

---

## Cau truc thu muc sau khi chay xong

```
artifacts/scale_paper/
├── id_label.json              # title / pv / label cho moi product
├── path_manifest.json         # duong dan anh / video
├── train_ids.json             # split train (~5953)
├── test_ids.json              # query eval
├── gallery_ids.json           # gallery eval
├── tsv_features/features.tsv  # region 36x2048
├── video_feature/*.npy        # S3D 12x1024
├── audio_feature/*.npy        # mel 12 frames
├── lmdb_features/             # train/test/gallery LMDB
├── checkpoints/scale_paper_simcl/pytorch_model_*.bin
├── features/{test,gallery}/   # tpiva embeddings sau pretrain
├── retrieval_results/         # rank list cho evaluate_unit_v2
├── evaluation_benchmark.json  # mAP/Prec@1/5/10 (paper protocol)
├── index_hnsw/                # Faiss cho API
└── pretrain.log
```

---

## Full pipeline (tung buoc)

Thay `Y:\SCALE` bang duong dan repo cua ban.

```powershell
$py = "C:\Users\SenetUser\AppData\Local\Programs\Python\Python312\python.exe"
$repo = "Y:\SCALE"
$env:PYTHONPATH = "$repo\app\SCALE;$repo\app"
Set-Location $repo
```

### Buoc 0 - Metadata M5Product (mot lan)

Can file metadata goc (khong commit vi lon):

- `app/datasets/product1m_product5m_id_label.json`

Tai tu [M5Product dataset](https://xiaodongsuper.github.io/M5Product_dataset/download.html).

---

### Buoc 1 - Download & chon subset can bang

Script: `app/datasets/download-dataset-tools.py`

```powershell
& $py app/datasets/download-dataset-tools.py `
  --input app/datasets/product1m_product5m_id_label.json `
  --output app/datasets/downloaded_m5product_balanced `
  --super-categories 50 `
  --per-super-category 200 `
  --workers 16 `
  --resume
```

Mặc định mỗi siêu danh mục được chia có kiểm soát: **70% đầy đủ** (image/title/PV/video/audio), **20% bị ẩn 1--2 modality** và **10% image-only**. Audio được tách từ video và phải được đối chiếu bằng `audio_feature_manifest.json`. Mọi cohort đều bắt đầu từ listing đủ nguồn modality; trường gốc và media được giữ để tái lập ablation. Không đổi seed hoặc các tỷ lệ nếu muốn so sánh trực tiếp giữa các lần chạy.

**Output:** `metadata.json`, `manifest.jsonl`, `selection_protocol.json`, `modality_selection.json`, ảnh/video đã tải.

Test nhanh:

```powershell
--max-products 1000
```

---

### Buoc 2 - Tao split train / val / test / gallery

Script: `app/preprocess/prepare_subset.py`

Split theo **label**: 70\% train, 10\% validation và test pool 20\% được tách thành 10\% query + 10\% gallery. Label có dưới 4 item bị loại vì không thể có train/val/query/gallery hợp lệ. Positive khi eval là cùng label **trong gallery**; đây không phải GT chính thức M5Product.

```powershell
& $py app/preprocess/prepare_subset.py `
  --dataset-dir app/datasets/downloaded_m5product_balanced `
  --output-dir artifacts/scale_paper_splits `
  --seed 42
```

**Output:** `train.json`, `val.json`, `test.json`, `gallery.json`, `records.json`, `split_protocol.json`.

---

### Buoc 3 - Convert sang layout SCALE

Script: `app/SCALE/preprocess/convert_balanced_to_scale.py`

```powershell
& $py app/SCALE/preprocess/convert_balanced_to_scale.py `
  --dataset-dir app/datasets/downloaded_m5product_balanced `
  --splits-dir artifacts/scale_paper_splits `
  --output-dir artifacts/scale_paper
```

---

### Buoc 4 - Extract sidecar features

#### 4a. Region features -> TSV (lau nhat, vai gio cho 10K)

```powershell
& $py app/SCALE/tools/bp_feature/extract/extract_regions_windows.py `
  --id-label artifacts/scale_paper/id_label.json `
  --path-manifest artifacts/scale_paper/path_manifest.json `
  --output-tsv artifacts/scale_paper/tsv_features/features.tsv `
  --backend auto
```

`--backend detectron2` neu da cai Detectron2; `auto` thu Detectron2 roi fallback torchvision.

#### 4b. Video (S3D) + audio mp3

```powershell
& $py app/SCALE/preprocess/extract_video_audio.py `
  --path-manifest artifacts/scale_paper/path_manifest.json `
  --video-output-dir artifacts/scale_paper/video_feature `
  --audio-output-dir artifacts/scale_paper/audios `
  --zero-fill-missing `
  --reextract-zero
```

#### 4c. Audio mel -> `.npy`

```powershell
& $py app/SCALE/tools/audio_process/save_audio_feature.py `
  --id-label artifacts/scale_paper/id_label.json `
  --audio-dir artifacts/scale_paper/audios `
  --output-dir artifacts/scale_paper/audio_feature `
  --zero-fill-missing
```

---

### Buoc 5 - Build LMDB (Windows: bat buoc `--map-size-gb 8`)

Tensorpack mac dinh 128 MB tren Windows -> **mat ~50% record** neu khong tang map size.

```powershell
foreach ($split in @("train","test","gallery")) {
  & $py app/SCALE/tools/bp_feature/convert/convert_from_config.py `
    --tsv-dir artifacts/scale_paper/tsv_features `
    --output-lmdb artifacts/scale_paper/lmdb_features/${split}_feature.lmdb `
    --ids-file artifacts/scale_paper/${split}_ids.json `
    --tsv-name features.tsv `
    --map-size-gb 8
}
```

**Verify:** train LMDB phai ~ **5953** entries (khong phai ~2958).

---

### Buoc 6 - Pretrain SIMCL (MLM+MRM+MEM+MFM+MAM+CLR)

Khuyen nghi: `scripts/start_pretrain.ps1` (log realtime, tranh chay trung process).

```powershell
powershell -NoProfile -File scripts/start_pretrain.ps1 -TrainEpochs 10
```

Theo doi log:

```powershell
Get-Content artifacts/scale_paper/pretrain.log -Wait -Tail 15
```

**Resume** tu checkpoint:

```powershell
powershell -NoProfile -File scripts/start_pretrain.ps1 `
  -TrainEpochs 10 `
  -StartEpoch 1 `
  -FromCheckpoint "Y:\SCALE\artifacts\scale_paper\checkpoints\scale_paper_simcl\pytorch_model_0.bin" `
  -AppendLog
```

| Tham so | Gia tri | Ghi chu |
|---------|---------|---------|
| train_batch_size | 16 | effective batch |
| gradient_accumulation_steps | 8 | micro-batch = 2 |
| num_train_epochs | 10 | paper dung `pytorch_model_9.bin` |
| FP16 | **tat** | Apex fp16 khong on tren Windows |

**Thoi gian uoc tinh (5953 train, GTX 1660 SUPER):**

| Epoch | ~Thoi gian |
|-------|------------|
| 1 | ~15 phut |
| 10 | ~2.5 gio |

Checkpoint: `artifacts/scale_paper/checkpoints/scale_paper_simcl/pytorch_model_{epoch}.bin` (~1 GB/epoch).

---

### Buoc 7 - Extract embedding + Evaluate + Faiss

Script: `app/SCALE/preprocess/resume_pipeline.py`

```powershell
& $py app/SCALE/preprocess/resume_pipeline.py `
  --work-dir artifacts/scale_paper `
  --skip-wait `
  --skip-lmdb `
  --skip-pretrain
```

Chi eval lai (feature da co):

```powershell
& $py app/SCALE/preprocess/resume_pipeline.py `
  --work-dir artifacts/scale_paper `
  --skip-wait --skip-lmdb --skip-pretrain --skip-extract
```

**Metric paper:** `evaluate_unit_v2` -> `evaluation_benchmark.json`

| Field | Y nghia |
|-------|---------|
| mAP | mAP@1 / @5 / @10 |
| Prec | Precision@1 / @5 / @10 |
| Feature type | `--eval-feature-types tpiva` (5 modality) |

So sanh paper: [M5Product Benchmark - SCALE tpiva](https://xiaodongsuper.github.io/M5Product_dataset/benchmark.html) (subset finetune mAP@10 ~ **71.5**).

---

### Buoc 8 - Chay API retrieval

```powershell
powershell -NoProfile -File scripts/start_backend.ps1
powershell -NoProfile -File scripts/start_frontend.ps1
```

Backend dung `SCALE_BACKEND=paper`, `SCALE_WORK_DIR=artifacts/scale_paper`, index Faiss trong `index_hnsw/`.

---

## One-shot script (tat ca phase)

Chay buoc 1-2 truoc, roi:

```powershell
powershell -NoProfile -File app/training/run_pipeline_scale.ps1 `
  -DatasetDir app/datasets/downloaded_m5product_balanced `
  -SplitsDir artifacts/scale_paper_splits `
  -WorkDir artifacts/scale_paper `
  -TrainEpochs 10 `
  -BatchSize 16 `
  -GradAccum 8
```

Flags huu ich:

| Flag | Y nghia |
|------|---------|
| `-PrepareOnly` | Chi convert layout (buoc 3) |
| `-SkipFeatures` | Bo extract + LMDB |
| `-SkipPretrain` | Bo pretrain |
| `-SkipEval` | Bo evaluate_unit_v2 |
| `-Limit 50` | Smoke test 50 san pham |
| `-SmokeTest` | `-Limit 50 -TrainEpochs 1` |

---

## Resume sau khi gian doan

| Tinh huong | Lenh |
|------------|------|
| Download do | `download-dataset-tools.py --resume` |
| Region TSV do | Chay lai `extract_regions_windows.py` (append) |
| Pretrain do | `start_pretrain.ps1 -StartEpoch N -FromCheckpoint ... -AppendLog` |
| Chi eval lai | `resume_pipeline.py --skip-pretrain --skip-extract ...` |
| Backup Google Drive | Sync ca `artifacts/scale_paper/` (~10-20 GB) |

---

## Kien truc pretrain (SIMCL)

```
Title, Table(pv), Image, Video, Audio
    -> embedding rieng -> encoder 6L/modality
    -> CLR (contrastive + DyCTR graph)
    -> JCT cross_encoder 6L
    -> MLM + MEM + MRM + MFM + MAM + CLR
```

- **SIMCL** = tong 6 loss tren (khong gom ITM).
- **CLR** = contrastive loss giua cac cap modality.

---

## Troubleshooting

| Trieu chung | Nguyen nhan | Cach sua |
|-------------|-------------|----------|
| LMDB train ~2958 thay vi ~5953 | map_size 128 MB tren Windows | Rebuild voi `--map-size-gb 8` |
| NaN loss, batch skip | CLR + modality thieu | Binh thuong rai rac; train tiep |
| FP16 crash | Apex tren Windows | Khong dung `--fp16` |
| mAP = 0 sau 1 epoch | Model chua hoc | Train >=10 epoch tren LMDB du |
| Metric thap hon paper | Thieu finetune, torchvision region, zero-fill video/audio | Xem muc cai thien them |

### Cai thien them (tuy chon, gan paper hon)

1. **Finetune:** `app/SCALE/examples/SCALE/train_cls.py` + `run_train_cls.sh` (can `label_list.json`)
2. **Detectron2 region** thay torchvision
3. **Video/audio** extract day du, giam zero-fill
4. **Finetune + eval** tren protocol M5Product official GT (neu co file GT day du)

---

## Scripts tham chieu nhanh

| Script | Muc dich |
|--------|----------|
| `app/datasets/download-dataset-tools.py` | Download + can bang subset |
| `app/preprocess/prepare_subset.py` | Split train/val/test/gallery |
| `app/SCALE/preprocess/convert_balanced_to_scale.py` | Layout SCALE |
| `app/SCALE/tools/bp_feature/extract/extract_regions_windows.py` | Region TSV |
| `app/SCALE/preprocess/extract_video_audio.py` | Video S3D |
| `app/SCALE/tools/audio_process/save_audio_feature.py` | Audio mel |
| `app/SCALE/tools/bp_feature/convert/convert_from_config.py` | LMDB |
| `scripts/start_pretrain.ps1` | Pretrain SIMCL |
| `app/SCALE/preprocess/resume_pipeline.py` | Extract + evaluate_unit_v2 + Faiss |
| `app/training/run_pipeline_scale.ps1` | Full pipeline (PowerShell) |
| `scripts/start_backend.ps1` | FastAPI paper backend |
| `scripts/start_frontend.ps1` | UI |
