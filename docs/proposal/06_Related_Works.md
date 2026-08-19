# 06. Related Works

## 6.1. Tổng quan

Visual product search tách thành hai lớp: **biểu diễn listing** và **truy hồi vector**. Các bài trong `docs/paper` cho thấy hầu hết hệ thống công nghiệp vẫn dừng ở ảnh, hoặc ảnh+title, hoặc **text query** + ảnh item. SCALE khác chỗ nhận **năm modality** (image, text, table, video, audio), học trọng số tương tác, và train cả mẫu thiếu. Đề tài giữ SCALE làm encoder, thêm Faiss HNSW và tái xếp hạng thuộc tính ở serving.

## 6.2. M5Product / SCALE (Dong et al., 2022)

**Bài toán.** Pretraining đa phương thức bị kẹt vì thiếu dataset lớn hơn image–text. E-commerce tự nhiên có caption, bảng spec, video bán hàng và audio. Hai thách thức: (1) interaction khi số modality ≥ 3; (2) noise do missing modality, long-tail.

**Dữ liệu.** M5Product: 6,313,067 mẫu, 6,232 category, 5,679 thuộc tính, crawl listing thật (CC BY-NC-SA 4.0). ~20% thiếu nhánh; ~5% unimodal. Split train 4.42M / 3,593 class; retrieval coarse và fine-grained; classification 1,805 class.

**Phương pháp.** Single-stream: encoder riêng từng modality → concat token → Joint Co-Transformer. Masked tasks 15%: MLM, MRP (region), MEM (cả entity table), MFP (frame), MAM (audio). SIMCL học ma trận `S` (init 0): `S ← S · softmax(S)`; tam giác nhân contrastive InfoNCE, đường chéo nhân masked loss. Missing = zero imputation.

**Kiến trúc/train.** Text init BERT; 6 layer unimodal + 6 layer fusion; hidden 768; caption ≤ 36, table ≤ 64; Faster R-CNN ResNet101 (Visual Genome), 10–36 box; audio MFCC. Paper: batch 64, 5 epoch, Adam `1e-4`. Code eval load `pytorch_model_9.bin` (epoch index 9).

**Kết quả.** Thêm modality thì Accuracy/mAP tăng, mức tăng lớn hơn trên full set. Subset 5 modality: Acc **85.50**, mAP@1 **58.72 / 70.62** (pretrain/finetune). I+T: SCALE 51.47 mAP@1 vs CAPTURE 50.30, UNITER 49.87.

**Gap với đề tài.** Paper đo exact inner product, không ANN serving, không rerank metadata trên top ứng viên.

## 6.3. Shopsy — Visually Similar Products Retrieval (Nadkarni & Dasararaju, 2022)

**Bối cảnh.** Reseller trên Facebook/WhatsApp gửi **ảnh** (không link). Ảnh bị nén chat, crop, scribble, logo reseller. Text search đẩy về head brand.

**Phương pháp.** Multi-task trên ảnh fashion Flipkart: (1) attribute classification có masking; (2) triplet ranking với **offline mining đa thuộc tính** (không chỉ một class label); (3) VAE chống nhiễu nén/crop. Production: embedding → ANN (HNSW/ScaNN), tối ưu Precision@K, QPS, memory, tần suất cập nhật catalog.

**Điểm mạnh.** Rõ ràng về nhiễu ảnh thật và lựa chọn index.

**Giới hạn.** Chỉ image-to-image, miền Lifestyle/fashion; không table/video/audio; không missing-modality pretraining.

## 6.4. MIEM — Transformer-Empowered Multi-Modal Item Embedding (Liu et al., AAAI 2024)

**Bài toán Shopee image search.** I2I thuần: (1) thiên texture/packaging (chỉ nha khoa kéo sạc pin vì vỏ giống); (2) nhiễu khi ảnh nhiều object; (3) mỗi ảnh một vector → index phình, phải map nhiều ảnh về một product.

**Hệ thống.** Query = ảnh user (detect box → crop). Hai recall song song: I2I và MIEM, merge ID, rồi rank. Offline precompute hai index HNSW.

**Mô hình.** Dual-tower. Query tower: Swin Transformer Base trên ảnh query. Item tower: nhiều ảnh sản phẩm + title (mBERT); **6 layer merge-attention** (concat patch + token, tốt hơn cross-attention cho recall). Chiếu 128 chiều. Train InfoNCE từ click log, từng country một model.

**Kết quả.** Deploy 3/2023: +9.90% clicks/user, +4.23% orders/user.

**Giới hạn.** Query chỉ ảnh; catalog chỉ ảnh+title; không table/video/audio; fusion item-side, không SIMCL năm nhánh; không zero-impute mẫu thiếu video.

## 6.5. MRSE — Multi-modality Retrieval for Large Scale E-commerce (Jiang et al., 2024)

**Bài toán.** Search **text query** trên Shopee. Uni-modality ERS (FastText/BERT) lệch vì query và title khác convention; không bắt được ý màu (“red guitar” vs dây đỏ).

**Kiến trúc.** Two-tower, hai giai đoạn. LMoE ba expert: **VBert** (ViT-L đóng băng + adaptor, rồi Light-BERT 2 layer/128-d) cho image–text; **Light-Bert** cho text; **FtAtt** (FastText + attention) cho lịch sử n-gram. DSSM fuse. User portrait đa modality từ lịch sử click (ảnh đã xem, query cũ) để chỉnh preference (quần áo thiên ảnh, cây lau nhà thiên mô tả chức năng).

**Loss.** Hybrid: in-batch softmax CE + triplet với hard-easy negative sampling — vì triplet/softmax đơn thuần khó hội tụ trên billions.

**Serving.** HNSW, cosine, two-tower late fusion để latency thấp.

**Kết quả.** +18.9% relevance offline, +3.7% core metric online; trở thành base model Shopee Search.

**Giới hạn.** Query là **câu text**, không phải ảnh/video. “Multi-modality” = query text + item image + user history, không phải năm nhánh listing như SCALE. Không MEM/table entity, không video pretraining.

## 6.6. FashionMV / ProCIR (Yuan et al., 2026)

**Bài toán.** Composed Image Retrieval (CIR): ảnh tham chiếu + câu sửa (“muốn kiểu hở lưng”). Mọi benchmark CIR (FashionIQ, CIRR, FACap) là **image-level**. Fashion có **View Incompleteness**: cổ V chỉ thấy từ trước, dây chéo sau lưng chỉ thấy từ sau — một ảnh không đủ.

**Dữ liệu.** FashionMV: 127K sản phẩm, 472K ảnh multi-view, 220K+ triplet CIR; pipeline tự động (Kimi caption → Qwen lọc ảo giác trái/phải → Gemini chọn target và viết modification). 99.83% triplet ≥ 2 viewpoint.

**Phương pháp.** ProCIR trên Qwen3.5-0.8B: mọi view nhét một request, token `<emb>` làm product embedding (d=1024). Ba cơ chế: two-stage dialogue (tách nhìn và lý giải câu sửa), caption-based alignment, CoT; SFT tùy chọn. Ablation 16 cấu hình: alignment là then chốt; two-stage là điều kiện; SFT và CoT trùng vai trò knowledge.

**Giới hạn.** Fashion CIR, query = ảnh + modification text, không table/video/audio e-commerce tổng quát; gallery là product-level fashion embedding, không M5Product.

## 6.7. So sánh theo yêu cầu đề tài

| Method | Interaction | Missing modality | Query | Index |
| --- | --- | --- | --- | --- |
| I2I / Shopsy | Không fusion 5 nhánh | Không | Ảnh (thường bị nén/crop) | HNSW/ScaNN |
| MIEM | Merge-attn ảnh×N + title | Không đặt video/table thiếu | Ảnh | HNSW, 128-d |
| MRSE | LMoE text–image–history | Không | **Text** | HNSW two-tower |
| FashionMV | MLLM gom multi-view + text sửa | Yêu cầu 2–5 view | Ảnh + câu sửa | Embedding product-level |
| **SCALE + HNSW + rerank (ours)** | JCT + SIMCL `S` | Zero imputation | `(I,T,Tab,V,A)`, tối thiểu I hoặc V | HNSW add tăng dần |

## 6.8. Vì sao chọn SCALE + HNSW

Listing e-commerce đúng như M5Product: năm tín hiệu, thường thiếu. MIEM/MRSE mạnh production nhưng không cover video/table/audio và missing-modality SSL. Shopsy/FashionMV mạnh nhiễu ảnh hoặc multi-view fashion, hẹp miền. SCALE đã có masked tasks năm nhánh và `S` tự học — khớp Mục 05.1–05.2.

HNSW (Malkov & Yashunin): add vector không train IVF, đồ thị đa tầng, cập nhật catalog không rebuild — đúng ràng buộc 05.4. Exact search của paper giữ làm protocol so Table 3–5; HNSW là lớp serving.

Tái xếp hạng thuộc tính (Mục 08) bù phần SCALE không làm: chữ đè ảnh và SKU gần giống sau khi đã có top-N.
