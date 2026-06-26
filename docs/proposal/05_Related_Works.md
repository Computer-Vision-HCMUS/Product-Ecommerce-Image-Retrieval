# 05. Related Works

## 5.1. Tổng quan

Các bài báo trong `docs/paper/related works` cho thấy visual product search đang phát triển theo ba hướng chính: học embedding đa phương thức, hiểu sản phẩm ở mức product-level/multi-view, và triển khai vector retrieval quy mô lớn. Từ đó, phương pháp của chúng tôi chọn **SCALE** để trích xuất đặc trưng đa phương thức và **Faiss-based ANN retrieval** để đánh chỉ mục/truy hồi nhanh.

## 5.2. Các công trình liên quan

| Paper | Ý tưởng chính | Điểm mạnh | Giới hạn/gap |
| --- | --- | --- | --- |
| M5Product/SCALE | Dataset 5 modality và Self-harmonized Contrastive Learning để học representation cho e-commerce. | Giải quyết modal diversity, missing modality, semantic alignment. | Chưa tập trung sâu vào production ANN serving. |
| Visually Similar Products Retrieval for Shopsy | Multi-task visual embedding với attribute classification, triplet ranking, VAE và ANN retrieval. | Có kinh nghiệm production: augmentation, PCA, ScaNN/HNSW, Precision@K/QPS. | Chủ yếu image-to-image, chưa khai thác đủ 5 modality. |
| Transformer-Empowered Multi-Modal Item Embedding | MIEM dùng ảnh nhiều góc và product title, kết hợp I2I + multi-modal item embedding tại Shopee. | Tăng semantic awareness và giảm storage so với index từng ảnh. | Phụ thuộc text/title và deployment từng marketplace. |
| MRSE | Multi-modality retrieval system cho large-scale e-commerce search, dùng text, image và user preference. | Tối ưu search công nghiệp với LMoE, hybrid loss, online metrics. | Tập trung nhiều vào text query/user behavior hơn image query thuần. |
| FashionMV | Product-level composed image retrieval với multi-view fashion data. | Nhấn mạnh view incompleteness và product-level representation. | Hẹp miền fashion và yêu cầu dữ liệu multi-view/caption phức tạp. |

## 5.3. Bảng so sánh theo gap

| Method | Sensory Gap | Semantic Gap | Context-Query Gap | Modal Gap |
| --- | --- | --- | --- | --- |
| Single image CNN/I2I | Trung bình: học texture/shape tốt nhưng dễ nhạy với crop, compression. | Yếu: dễ trả về sản phẩm nhìn giống nhưng sai category. | Yếu: không hiểu intent ngoài ảnh. | Yếu: chỉ image. |
| Triplet + attribute + VAE (Shopsy) | Tốt: augmentation và VAE giúp robust hơn. | Trung bình: attribute/triplet giảm sai fine-grained. | Trung bình: tốt cho image query thực tế nhưng ít context modality khác. | Yếu-Trung bình: chủ yếu image và attribute. |
| MIEM | Trung bình-Tốt: nhiều ảnh sản phẩm giảm phụ thuộc một view. | Tốt: title + image giúp hiểu category/semantic. | Trung bình: query vẫn chủ yếu image, context đến từ product title. | Trung bình: image + text. |
| MRSE | Trung bình: có image feature nhưng không tối ưu riêng cho noisy image query. | Tốt: text, image, user preference được align. | Tốt: modeling preference theo user/history. | Trung bình-Tốt: nhiều modality nhưng không đủ 5 modality của M5Product. |
| FashionMV/ProCIR | Tốt: multi-view giảm view incompleteness. | Tốt: product-level embedding và modification text. | Tốt: composed query image + text. | Trung bình: chủ yếu image + text trong fashion. |
| **SCALE + Faiss HNSW/IVF-PQ (ours)** | **Tốt**: tận dụng image/video/table/text/audio và có thể bổ sung augmentation. | **Tốt**: SCALE align semantic giữa modality, table/text bổ sung category/attribute. | **Tốt**: query nhiều modality, có thể biểu diễn intent bằng text/table/audio. | **Rất tốt**: thiết kế trực tiếp cho 5 modality và missing modality. |

## 5.4. Vì sao lựa chọn này phù hợp với Product E-commerce Retrieval

Product E-commerce Retrieval có hai yêu cầu khác nhau nhưng phải giải quyết đồng thời. Thứ nhất là **biểu diễn đúng sản phẩm**: hệ thống phải hiểu ảnh, tiêu đề, thuộc tính bảng, video và audio trong cùng một không gian semantic. Thứ hai là **truy hồi nhanh ở quy mô catalog lớn**: sau khi có embedding, hệ thống phải tìm top-K trong hàng trăm nghìn đến hàng triệu sản phẩm mà không duyệt tuyến tính toàn bộ catalog ở mỗi query. Vì vậy, phương pháp được chọn cần tách rõ **feature extraction** và **retrieval indexing**.

### 5.4.1. Vì sao dùng SCALE cho feature extraction?

SCALE phù hợp hơn các hướng image-only vì dữ liệu e-commerce không chỉ nằm trong ảnh. Một ảnh sản phẩm có thể cho biết màu, hình dạng, texture; nhưng title và table lại cho biết brand, material, category, usage scene; video cho biết nhiều góc nhìn; audio hoặc voice có thể mang thông tin mô tả bổ sung. Các nguồn này giúp giảm bốn gap chính:

| Gap | SCALE hỗ trợ như thế nào? |
| --- | --- |
| Sensory Gap | Image/video branch giúp học nhiều góc nhìn; augmentation có thể bổ sung khả năng chống crop, compression, watermark. |
| Semantic Gap | Text/table branch bổ sung category, brand, material, function để tránh trả về sản phẩm chỉ giống texture nhưng sai ý nghĩa. |
| Context-Query Gap | Query có thể là image + text/table/audio, giúp mô hình hiểu intent tốt hơn image-only. |
| Modal Gap | SCALE được thiết kế trực tiếp cho 5 modality và có SIMCL để học alignment giữa các modality. |

Điểm mạnh quan trọng của SCALE là **Self-harmonized Inter-Modality Contrastive Learning (SIMCL)**. Với 5 modality, không phải cặp modality nào cũng hữu ích như nhau. Ví dụ image-text thường mạnh trong retrieval, image-audio có thể chỉ hữu ích với một số category. SIMCL học trọng số alignment giữa các modality, nhờ đó mô hình không ép mọi modality đóng góp ngang nhau. Điều này phù hợp với catalog e-commerce vì dữ liệu thường thiếu modality, nhiễu và long-tail.

### 5.4.2. Vì sao cần FlatL2/FlatIP baseline?

FlatL2/FlatIP không phải lựa chọn triển khai cuối cho catalog lớn, nhưng nó rất cần trong nghiên cứu vì đây là **exact search baseline**. Nếu HNSW hoặc IVF-PQ trả kết quả kém, ta cần biết lỗi đến từ embedding SCALE hay từ approximation của index.

Flat baseline giúp trả lời ba câu hỏi:

- Embedding của SCALE có đủ phân biệt sản phẩm không?
- ANN index làm mất bao nhiêu recall so với exact search?
- Khi tune HNSW/IVF-PQ, cấu hình nào đạt trade-off tốt nhất giữa speed và accuracy?

Vì vậy, FlatL2/FlatIP là mốc đo chất lượng bắt buộc trước khi kết luận Faiss HNSW hoặc IVF-PQ tốt.

### 5.4.3. Vì sao dùng Faiss HNSW làm index chính?

Faiss HNSW phù hợp làm index chính cho prototype vì:

- Không cần training index như IVF/PQ, nên dễ xây pipeline ban đầu.
- Có recall-latency trade-off tốt thông qua `efSearch`, `efConstruction`, `M`.
- Phù hợp với embedding dense như output của SCALE.
- Dễ so sánh với FlatL2/FlatIP trong cùng thư viện Faiss.
- Phù hợp với yêu cầu demo: cần top-K nhanh, không nhất thiết phải tối ưu memory ngay từ đầu.

Trong Product E-commerce Retrieval, người dùng thường chỉ nhìn top vài kết quả đầu. Vì vậy, index chính cần ưu tiên Precision@K/Recall@K ở K nhỏ và latency thấp. HNSW là lựa chọn cân bằng tốt cho giai đoạn này.

### 5.4.4. Vì sao cần IVF-PQ/OPQ-PQ?

HNSW nhanh và chính xác nhưng tốn RAM vì lưu graph links và vector. Khi catalog tăng lên hàng triệu hoặc hàng chục triệu sản phẩm, memory footprint trở thành bottleneck. IVF-PQ/OPQ-PQ phù hợp làm phương án mở rộng vì:

- IVF chia vector space thành nhiều cụm, lúc search chỉ duyệt một số cụm gần query.
- PQ nén vector thành code ngắn hơn, giảm memory.
- OPQ xoay không gian vector để PQ nén hiệu quả hơn.

Đổi lại, IVF-PQ có thể làm giảm precision/recall và cần training/tuning (`nlist`, `nprobe`, code size). Vì vậy proposal không chọn IVF-PQ làm index đầu tiên, mà dùng nó như phương án scale khi HNSW bắt đầu quá nặng.

### 5.4.5. Vì sao ScaNN chỉ là optional benchmark?

Paper Shopsy cho thấy ScaNN có QPS rất tốt và Precision@4 tương đương các index mạnh trong bối cảnh production. Tuy nhiên, trong đồ án này ScaNN nên là optional benchmark thay vì dependency chính vì:

- Môi trường cài đặt có thể khó hơn Faiss.
- Faiss đã đủ để xây exact baseline, HNSW và IVF-PQ trong cùng một toolkit.
- Mục tiêu chính của đề tài là chứng minh pipeline SCALE + vector retrieval, không phụ thuộc vào một thư viện search duy nhất.

Do đó, ScaNN được dùng để đối chiếu nếu còn thời gian và môi trường hỗ trợ. Nếu ScaNN không chạy được, hệ thống vẫn hoàn chỉnh với FlatL2/FlatIP + Faiss HNSW + Faiss IVF-PQ.

### 5.4.6. Tóm tắt vai trò từng thành phần

| Thành phần | Vai trò trong proposal | Lý do cần có |
| --- | --- | --- |
| SCALE | Multi-modal feature extractor. | Giải quyết semantic/modal gap bằng image, text, table, video, audio. |
| FlatL2/FlatIP | Exact baseline. | Đo chất lượng embedding và đo recall loss của approximate index. |
| Faiss HNSW | Index chính. | Dễ triển khai, không cần train, recall-latency tốt cho demo. |
| Faiss IVF-PQ/OPQ-PQ | Index scale/nén. | Giảm memory khi catalog lớn. |
| ScaNN | Optional benchmark. | Có bằng chứng tốt từ Shopsy, nhưng không bắt buộc để hoàn thành hệ thống. |

Như vậy, lựa chọn này phù hợp với Product E-commerce Retrieval vì nó không chỉ tối ưu một phía. SCALE tập trung vào chất lượng biểu diễn sản phẩm, Flat baseline giúp đánh giá khoa học, Faiss HNSW phục vụ demo/latency, IVF-PQ phục vụ scale, và ScaNN giúp so sánh với hướng production trong related work.

## 5.5. Kết luận lựa chọn phương pháp

SCALE phù hợp với feature extraction vì nó được thiết kế cho M5Product, học adaptive modality importance và xử lý missing modality. Với retrieval layer, chúng tôi không xem ANN là một index cụ thể mà là nhóm kỹ thuật cần chọn implementation. Proposal dùng **FlatL2/FlatIP** làm exact baseline, **Faiss HNSW** làm index chính vì cân bằng tốt giữa recall và latency, **Faiss IVF-PQ/OPQ-PQ** khi memory là bottleneck, và **ScaNN** là optional benchmark vì Shopsy cho thấy ScaNN/HNSW hoạt động tốt trong môi trường production. Các lựa chọn này cần được so sánh bằng Precision@K, Recall@K, QPS, memory footprint và build time.
