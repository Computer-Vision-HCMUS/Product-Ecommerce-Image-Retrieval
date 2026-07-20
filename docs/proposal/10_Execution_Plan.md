# 10. Execution Plan

## 10.1. Thời gian thực hiện

Kế hoạch kéo dài 2 tháng, chia thành 8 tuần.

| Tuần | Công việc | Kết quả |
| --- | --- | --- |
| Week 1 | Đọc paper, chốt problem statement, chuẩn hóa proposal, khảo sát dataset M5Product. | Proposal hoàn chỉnh, danh sách requirement và metric. |
| Week 2 | Chuẩn bị data loader, preprocess image/text/table/video/audio, thiết kế schema metadata. | Pipeline đọc dữ liệu và kiểm tra sample. |
| Week 3 | Cài đặt hoặc tái hiện SCALE feature extraction; chạy thử trên subset nhỏ và kiểm tra region proposal. | Embedding extraction chạy được; có thống kê region recall/region failure cơ bản. |
| Week 4 | Pretrain/finetune thử nghiệm và tạo failure slice cho ảnh nhiễu/nhiều object. | Checkpoint đầu tiên, log training và bộ kiểm tra robustness/region failure. |
| Week 5 | Export gallery/query embeddings, xây `IndexFlatIP` baseline và Faiss HNSW index. | Index đầu tiên, kết quả Precision@K/Recall@K baseline. |
| Week 6 | Tune Faiss HNSW và Faiss IVF-PQ/OPQ-PQ với `IndexFlatIP` làm exact baseline; thử exact re-ranking trên Top-N. | Bảng Recall@K, latency, QPS, memory, build time và ablation re-ranking. |
| Week 7 | Xây demo/API retrieval, visualize top-K result, thêm logging failure cases và attribute-aware re-ranking cho query có context đáng tin cậy. | Demo end-to-end và báo cáo cải thiện theo failure slice. |
| Week 8 | Tổng hợp kết quả, viết báo cáo, hoàn thiện slide/demo, phân tích hạn chế. | Final report, demo, evaluation table. |

## 10.2. Milestones

| Milestone | Deadline | Deliverable |
| --- | --- | --- |
| M1 | Cuối Week 1 | Proposal và scope hoàn chỉnh. |
| M2 | Cuối Week 3 | Data + feature extraction prototype. |
| M3 | Cuối Week 5 | Retrieval baseline với `IndexFlatIP` và Faiss HNSW. |
| M4 | Cuối Week 7 | Demo end-to-end và ablation cho các cải thiện đã chọn. |
| M5 | Cuối Week 8 | Báo cáo cuối và kết quả đánh giá. |

## 10.3. Phân công dự kiến

| Thành viên | Trọng tâm |
| --- | --- |
| Trần Hải Đức | Feature extraction, SCALE, preprocessing, evaluation metrics. |
| Trần Hoàng Nam | Faiss HNSW/IVF-PQ index, API/demo retrieval, benchmark latency/QPS, report visualization. |

## 10.4. Rủi ro kế hoạch

| Rủi ro | Ảnh hưởng | Phương án dự phòng |
| --- | --- | --- |
| M5Product quá lớn hoặc khó tải đầy đủ | Chậm training và storage cao. | Dùng subset theo category, ưu tiên image/text/table trước. |
| GPU hạn chế | Không train full SCALE được. | Dùng pretrained/finetune nhỏ, freeze backbone, giảm batch size. |
| Label retrieval không đủ | Metric yếu. | Dùng category label, instance label hoặc human-labeled subset nhỏ. |
| Detector bỏ sót sản phẩm long-tail | Embedding không nhận được đúng region cần truy hồi. | Đo region recall theo category và ghi nhận failure case. |
| Demo latency cao | Trải nghiệm demo kém. | Cache embedding, batch offline và tối ưu batch size. |
