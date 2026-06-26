# 07. Expected Results and Evaluation

## 7.1. Kết quả mong muốn

Hệ thống kỳ vọng đạt được:

- Truy hồi top-K ảnh sản phẩm tương đồng với query đa phương thức.
- Embedding thể hiện tốt cả visual similarity và semantic similarity.
- Faiss HNSW/IVF-PQ cho tốc độ truy hồi nhanh hơn exhaustive search đáng kể, trong khi FlatL2/FlatIP giữ vai trò exact baseline.
- Khả năng chống nhiễu với query bị crop, compression, rotation hoặc watermark.
- Pipeline có thể mở rộng cho catalog lớn và cập nhật sản phẩm định kỳ.

## 7.2. Tiêu chí đánh giá chất lượng mô hình

| Metric | Ý nghĩa | Lý do dùng |
| --- | --- | --- |
| Precision@K | Tỷ lệ kết quả đúng trong top-K. | Phù hợp với mục tiêu trả về ít kết quả nhưng chính xác. |
| Recall@K | Tỷ lệ ground-truth được tìm thấy trong top-K. | Đo khả năng không bỏ sót sản phẩm đúng. |
| mAP@K | Trung bình average precision theo nhiều query. | Đánh giá cả đúng/sai và thứ tự ranking. |
| NDCG@K | Đánh giá ranking khi có nhiều mức relevance. | Hữu ích nếu có label same/similar/irrelevant. |
| Category Accuracy | Tỷ lệ kết quả đúng category. | Đo semantic alignment. |
| Robustness by augmentation | Precision@K theo từng loại noise. | Kiểm tra sensory gap. |

## 7.3. Tiêu chí đánh giá hệ thống retrieval

| Metric | Ý nghĩa |
| --- | --- |
| Query latency | Thời gian từ lúc gửi query đến lúc nhận top-K. |
| QPS | Số query xử lý mỗi giây. |
| Index build time | Thời gian tạo index từ gallery embeddings. |
| Memory footprint | Bộ nhớ index cần dùng. |
| Recall loss vs FlatL2/FlatIP | Mức giảm chất lượng khi dùng HNSW/IVF-PQ thay exact search. |
| Update cost | Chi phí thêm sản phẩm mới vào index. |

## 7.4. Tiêu chí đánh giá sản phẩm so với sản phẩm khác

| Tiêu chí | Sản phẩm của chúng tôi | Baseline/sản phẩm khác |
| --- | --- | --- |
| Query modality | Image, text, video, audio, table. | Thường chỉ image hoặc image+text. |
| Semantic awareness | Dựa trên SCALE multi-modal embedding. | I2I thuần dễ thiên texture/shape. |
| Large-scale retrieval | FlatL2/FlatIP baseline, Faiss HNSW index chính, IVF-PQ khi cần giảm memory. | Exact search chậm, prototype index thiếu benchmark. |
| Robustness | Có augmentation và missing modality handling. | Dễ giảm chất lượng khi query nhiễu. |
| Evaluation | Kết hợp model metrics và system metrics. | Nhiều demo chỉ đánh giá qualitative. |
| Explainability | Có thể phân tích theo modality contribution và failure cases. | Khó biết sai do image, text hay index. |

## 7.5. Target kỳ vọng ban đầu

Các target này là mục tiêu thực nghiệm, không phải cam kết cuối:

- Precision@5 và Recall@10 cao hơn baseline image-only.
- HNSW/IVF-PQ recall so với FlatL2/FlatIP giữ ở mức chấp nhận được, ưu tiên mất ít hơn 3-5 điểm phần trăm.
- Query latency đủ thấp cho demo interactive.
- Faiss HNSW có QPS cao hơn FlatL2/FlatIP rõ rệt trên cùng tập gallery; ScaNN chỉ là benchmark bổ sung nếu môi trường hỗ trợ.

## 7.6. Failure analysis dự kiến

Hệ thống sẽ ghi nhận các nhóm lỗi:

- Same color/texture nhưng sai category.
- Cùng category nhưng sai model/variant.
- Query chứa nhiều object, chọn nhầm object chính.
- Product mới hoặc long-tail category chưa học tốt.
- Kết quả đúng semantic nhưng ảnh không giống trực quan.

Các lỗi này sẽ được dùng để điều chỉnh augmentation, finetuning data, modality weighting và index/rerank strategy.
