# 08. Execution Plan

## 8.1. Thời gian thực hiện

Kế hoạch kéo dài 2 tháng, chia thành 8 tuần.

| Tuần | Công việc | Kết quả |
| --- | --- | --- |
| Week 1 | Đọc paper, chốt problem statement, chuẩn hóa proposal, khảo sát dataset M5Product. | Proposal hoàn chỉnh, danh sách requirement và metric. |
| Week 2 | Chuẩn bị data loader, preprocess image/text/table/video/audio, thiết kế schema metadata. | Pipeline đọc dữ liệu và kiểm tra sample. |
| Week 3 | Cài đặt hoặc tái hiện SCALE feature extraction; chạy thử trên subset nhỏ. | Embedding extraction chạy được trên subset. |
| Week 4 | Pretrain/finetune thử nghiệm, thêm augmentation cho image query. | Checkpoint đầu tiên và log training. |
| Week 5 | Export gallery/query embeddings, xây FlatL2/FlatIP baseline và Faiss HNSW index. | Index đầu tiên, kết quả Precision@K/Recall@K baseline. |
| Week 6 | Tune retrieval index, so sánh FlatL2/FlatIP, Faiss HNSW, Faiss IVF-PQ/OPQ-PQ và ScaNN nếu có. | Bảng trade-off Precision, Recall, QPS, memory, build time. |
| Week 7 | Xây demo/API retrieval, visualize top-K result, thêm logging failure cases. | Demo end-to-end. |
| Week 8 | Tổng hợp kết quả, viết báo cáo, hoàn thiện slide/demo, phân tích hạn chế. | Final report, demo, evaluation table. |

## 8.2. Milestones

| Milestone | Deadline | Deliverable |
| --- | --- | --- |
| M1 | Cuối Week 1 | Proposal và scope hoàn chỉnh. |
| M2 | Cuối Week 3 | Data + feature extraction prototype. |
| M3 | Cuối Week 5 | Retrieval baseline với FlatL2/FlatIP và Faiss HNSW. |
| M4 | Cuối Week 7 | Demo end-to-end. |
| M5 | Cuối Week 8 | Báo cáo cuối và kết quả đánh giá. |

## 8.3. Phân công dự kiến

| Thành viên | Trọng tâm |
| --- | --- |
| Trần Hải Đức | Feature extraction, SCALE, preprocessing, evaluation metrics. |
| Trần Hoàng Nam | Faiss HNSW/IVF-PQ index, API/demo retrieval, benchmark latency/QPS, report visualization. |

## 8.4. Rủi ro kế hoạch

| Rủi ro | Ảnh hưởng | Phương án dự phòng |
| --- | --- | --- |
| M5Product quá lớn hoặc khó tải đầy đủ | Chậm training và storage cao. | Dùng subset theo category, ưu tiên image/text/table trước. |
| GPU hạn chế | Không train full SCALE được. | Dùng pretrained/finetune nhỏ, freeze backbone, giảm batch size. |
| ScaNN không tương thích môi trường | Không benchmark được ScaNN. | Dùng Faiss HNSW làm chính và Faiss IVF-PQ khi cần giảm memory. |
| Label retrieval không đủ | Metric yếu. | Dùng category label, instance label hoặc human-labeled subset nhỏ. |
| Demo latency cao | Trải nghiệm demo kém. | Cache embedding, batch offline, giảm dimension/PCA. |
