# 09. Kết quả mong muốn và hướng phát triển

## 9.1. Hệ thống kỳ vọng đạt được

- Truy vấn trả về top-K sản phẩm tương đồng về **vẻ ngoài và danh mục** với query đa phương thức `q = (Image, Text, Table, Video, Audio)`.
- Pretrain SCALE **10 epoch** (checkpoint `pytorch_model_9.bin`, `epochId` 0–9), rồi đo retrieval bằng **cùng protocol với bài báo / source gốc**, để so với số liệu Table 3–5.
- Pipeline mở rộng được cho catalog lớn và cập nhật listing định kỳ nhờ HNSW add tăng dần.
- Tầng tái xếp hạng (khi bật) cải thiện đúng đắn ngữ nghĩa so với chỉ `S_emb`.

Paper ghi train 5 epoch; script đánh giá gốc (`eval_gallery1_*.sh`) load `pytorch_model_9.bin`. Đề tài bám convention của code: train 10 epoch và so với kết quả gốc trên bảng paper.

## 9.2. Protocol đánh giá để so với bài báo

Không dùng protocol SigLIP + Faiss trên `downloaded_2k` khi so với số paper. So sánh gốc đi theo pipeline SCALE:

1. Extract pooled embedding từng modality (và tổng `t+p+i+v+a` / `dense` sau finetune CLS).
2. Ranking exact: `score = query · gallery^T`, loại chính query khỏi danh sách, `max_topk = 10`.
3. Ground truth: **cùng category label** (`evaluate_unit_v2.py`). Positive set là mọi ID cùng label.
4. Báo cáo **mAP@1,5,10** và **Prec@1,5,10**. Classification (nếu finetune): Accuracy trên test CLS.

Công thức AP trong code gốc: cộng precision tại mỗi hit rồi chia cho **số hit trong top-k** (không chia cho `min(|positives|, k)`). `k` thực tế bị cắt `min(K, |pos_set|, |rank_list|)`. Khi so với Table 3–5 phải giữ đúng công thức này.

| Task | Script gốc | Metric paper |
| --- | --- | --- |
| Retrieval sau pretrain | `extract_features.py` → `retrieval_unit_id_list_v2.py` → `evaluate_unit_v2.py` | mAP@K, Prec@K (pretrain / finetune) |
| Classification | `train_cls.py` (Accuracy mỗi epoch) | Accuracy |
| Retrieval sau CLS | `extract_features_cls.py` (`dense`) → cùng retrieval + eval | mAP@K, Prec@K |
| Clustering | NMI, Purity trên fused embedding | NMI, Purity |

Ablation modality dùng đúng tổ hợp paper: I+T, I+T+Tab, I+T+Tab+V, I+T+Tab+V+A, v.v.

## 9.3. Số liệu gốc cần đối chiếu

Bảng dưới lấy từ SCALE trên M5Product. Cặp `a / b` là **pretrain / finetune**. Mục tiêu đề tài: chạy 10 epoch trên subset, báo cáo cùng cột, ghi **delta** so với hàng tương ứng.

### Classification + retrieval khi thêm dần modality (Table 3)

**Subset paper**

| Modality | Accuracy | mAP@1 | mAP@5 | mAP@10 | Prec@1 | Prec@5 | Prec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Text | 77.42 | 47.70 / 65.10 | 53.63 / 68.39 | 51.59 / 66.99 | 47.70 / 65.10 | 30.96 / 44.89 | 24.15 / 33.44 |
| +Image | 79.58 | 51.47 / 67.02 | 56.16 / 69.85 | 54.41 / 68.43 | 51.47 / 67.02 | 33.41 / 46.29 | 25.55 / 34.29 |
| +Table | 82.83 | 57.14 / 67.97 | 61.71 / 70.34 | 59.64 / 69.38 | 57.14 / 67.97 | 38.02 / 46.85 | 28.99 / 34.36 |
| +Video | 84.31 | 58.57 / 69.79 | 63.15 / 72.30 | 61.02 / 70.67 | 58.57 / 69.79 | 39.26 / 47.44 | 29.56 / 34.78 |
| **+Audio (5 modality)** | **85.50** | **58.72 / 70.62** | **63.17 / 73.02** | **61.05 / 71.50** | **58.72 / 70.62** | **39.66 / 48.20** | **30.32 / 35.35** |

**Toàn bộ M5Product** (tham chiếu; đề tài không train full 6.3M)

| Modality | Accuracy | mAP@1 | mAP@5 | mAP@10 | Prec@1 | Prec@5 | Prec@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **+Audio (5 modality)** | **86.57** | **63.56 / 73.77** | **67.51 / 76.17** | **65.39 / 74.73** | **63.56 / 74.01** | **42.68 / 50.78** | **32.17 / 37.42** |

### Image+Text so baseline (Table 5, subset)

| Method | mAP@1 | Accuracy | NMI | Purity |
| --- | --- | --- | --- | --- |
| Imagebased | 15.17 | 27.67 | 63.62 | 54.86 |
| BERT | 47.70 | 77.42 | 76.35 | 68.80 |
| UNITER | 49.87 | 78.54 | 82.71 | 73.58 |
| CAPTURE | 50.30 | 78.69 | 83.06 | 74.14 |
| **SCALE (gốc)** | **51.47** | **79.58** | **84.23** | **75.81** |

Kỳ vọng trên subset 10k của nhóm: **cùng thứ tự xếp hạng** (5 modality > I+T+Tab > I+T; SCALE I+T ≥ BERT/image-only). Số tuyệt đối có thể thấp hơn Table 3 vì ít mẫu hơn subset paper; báo cáo rõ split và epoch, không gộp với số full-set.

## 9.4. Cải tiến HNSW + tái xếp hạng (ngoài protocol paper)

Sau khi có embedding SCALE 10 epoch, đo thêm trên Faiss HNSW và Mục 08. Các metric này **không có trong paper**, dùng để so baseline nội bộ (trước/sau rerank), không thay số Table 3–5.

Kỳ vọng gia tăng Prec@K, AP, mAP ở:

- Ablation **Ảnh–Text–Video** hoặc **Ảnh–Table–Video**.
- Tổng thể đủ modality khả dụng.
- Trước/sau tái xếp hạng; tách query có text (Hướng 1) và query chỉ ảnh (Hướng 2).

| Metric | Ý nghĩa | Khi nào dùng |
| --- | --- | --- |
| mAP@K, Prec@K (code gốc) | So với Table 3–5. | Bắt buộc. |
| Precision@K, Recall@K, nDCG@K | Protocol nhóm (loại self, chia AP cho `min(\|pos\|, k)`). | Báo cáo phụ; ghi rõ khác công thức paper. |
| Latency, QPS, add-time HNSW | Catalog động. | Hệ thống, không phải paper. |

## 9.5. Hướng phát triển tương lai

- Lọc nhiễu rác trên ảnh (chữ quảng cáo, logo, watermark).
- Phân đoạn ảnh, tách vùng sản phẩm trọng tâm trước khi đưa vào SCALE.

Hai hướng này nhắm Thách thức “nhiễu trên ảnh” ở Mục 05, nằm ngoài phạm vi so với kết quả gốc.

## 9.6. Failure analysis dự kiến

- Cùng màu/texture nhưng sai category (kéo mAP/Prec so với Table 3).
- Cùng ngành nhưng sai model/thương hiệu (fine-grained).
- Ảnh nhiều object hoặc chữ chiếm nửa khung.
- Hướng 2: đa số N ứng viên sai ngành → `SD*`/`D*` sai.
- Subset 10k / long-tail: Accuracy và mAP thấp hơn hàng subset paper.
