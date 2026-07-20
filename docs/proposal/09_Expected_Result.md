# 09. Expected Results and Evaluation

## 9.1. Kết quả mong muốn

Hệ thống kỳ vọng đạt được:

- Truy hồi top-K product entry phù hợp với ảnh query, kèm metadata và ảnh đại diện.
- Embedding thể hiện tốt cả visual similarity và semantic similarity.
- Faiss HNSW/IVF-PQ cho tốc độ truy hồi nhanh hơn exhaustive search đáng kể, trong khi `IndexFlatIP` giữ vai trò exact baseline.
- Khả năng chống nhiễu với query bị crop, compression, rotation hoặc watermark.
- Pipeline có thể mở rộng cho catalog lớn và cập nhật sản phẩm định kỳ.

## 9.2. Tiêu chí đánh giá chất lượng mô hình

Paper M5Product/SCALE báo cáo retrieval bằng mAP và Precision. Recall@K, NDCG@K, Category Match@K và các metric hệ thống dưới đây là protocol đánh giá do nhóm bổ sung cho pipeline ảnh-query + Faiss; chúng không phải cấu hình downstream gốc của paper.

| Metric | Ý nghĩa | Lý do dùng |
| --- | --- | --- |
| Precision@K | Tỷ lệ kết quả đúng trong top-K. | Phù hợp với mục tiêu trả về ít kết quả nhưng chính xác. |
| Recall@K | Tỷ lệ ground-truth được tìm thấy trong top-K. | Đo khả năng không bỏ sót sản phẩm đúng. |
| mAP@K | Trung bình average precision theo nhiều query. | Đánh giá cả đúng/sai và thứ tự ranking. |
| NDCG@K | Đánh giá ranking khi có nhiều mức relevance. | Hữu ích nếu có label same/similar/irrelevant. |
| Category Match@K | Tỷ lệ query có ít nhất một kết quả cùng category trong top-K. | Chỉ số bổ sung cho semantic discovery; không thay thế SKU/instance-level metric. |
| Robustness by noise slice | Precision@K theo từng loại noise. | Kiểm tra sensory gap. |

## 9.3. Tiêu chí đánh giá hệ thống retrieval

| Metric | Ý nghĩa |
| --- | --- |
| Query latency | Thời gian từ lúc gửi query đến lúc nhận top-K. |
| QPS | Số query xử lý mỗi giây. |
| Index build time | Thời gian tạo index từ gallery embeddings. |
| Memory footprint | Bộ nhớ index cần dùng. |
| Recall loss vs `IndexFlatIP` | Mức giảm chất lượng khi dùng HNSW/IVF-PQ thay exact search. |
| Update cost | Chi phí thêm sản phẩm mới vào index. |

## 9.4. Tiêu chí đánh giá sản phẩm so với sản phẩm khác

| Tiêu chí | Sản phẩm của chúng tôi | Baseline/sản phẩm khác |
| --- | --- | --- |
| Query modality | Image; có thể kèm text/table ngắn. Video/audio dùng như modality của catalog khi dữ liệu có sẵn. | Thường chỉ image hoặc image+text. |
| Semantic awareness | Dựa trên SCALE multi-modal embedding. | I2I thuần dễ thiên texture/shape. |
| Large-scale retrieval | `IndexFlatIP` baseline, Faiss HNSW index chính, IVF-PQ khi cần giảm memory. | Exact search chậm, prototype index thiếu benchmark. |
| Robustness | Báo cáo theo từng nhóm query nhiễu và missing modality. | Dễ giảm chất lượng khi query nhiễu. |
| Evaluation | Kết hợp model metrics và system metrics. | Nhiều demo chỉ đánh giá qualitative. |
| Failure analysis | Phân tích theo nhiễu ảnh, category, missing metadata, region proposal và loại index. | Nhiều demo chỉ đánh giá qualitative. |

## 9.5. Target kỳ vọng ban đầu

Các target này là mục tiêu thực nghiệm, không phải cam kết cuối:

- Precision@5 và Recall@10 cao hơn baseline image-only.
- Chọn cấu hình HNSW/IVF-PQ có recall loss so với `IndexFlatIP` phù hợp với latency và memory budget đã công bố.
- Query latency đủ thấp cho demo interactive.
- Benchmark `IndexFlatIP`, HNSW và IVF-PQ trên cùng gallery, cùng hardware và nhiều quy mô catalog.

## 9.6. Failure analysis dự kiến

Hệ thống sẽ ghi nhận các nhóm lỗi:

- Same color/texture nhưng sai category.
- Cùng category nhưng sai model/variant.
- Query chứa nhiều object, chọn nhầm object chính.
- Product mới hoặc long-tail category chưa học tốt.
- Kết quả đúng semantic nhưng ảnh không giống trực quan.

Các lỗi này sẽ được dùng để phân tích giới hạn của representation, modality và index/rerank strategy.
