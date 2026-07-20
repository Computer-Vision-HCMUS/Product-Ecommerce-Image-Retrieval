# 06. Related Works

## 6.1. Tổng quan

Các bài báo trong `docs/paper/related works` cho thấy visual product search đang phát triển theo ba hướng chính: học embedding đa phương thức, hiểu sản phẩm ở mức product-level/multi-view, và triển khai vector retrieval quy mô lớn. Từ đó, phương pháp của chúng tôi chọn **SCALE** để trích xuất đặc trưng đa phương thức và **Faiss-based ANN retrieval** để đánh chỉ mục/truy hồi nhanh.

## 6.2. Các công trình liên quan

| Paper | Ý tưởng chính | Điểm mạnh | Giới hạn/gap |
| --- | --- | --- | --- |
| M5Product/SCALE | Dataset 5 modality và Self-harmonized Contrastive Learning để học representation cho e-commerce. | Giải quyết modal diversity, missing modality, semantic alignment. | Chưa tập trung sâu vào production ANN serving. |
| Visually Similar Products Retrieval for Shopsy | Multi-task visual embedding với attribute classification, triplet ranking, VAE và ANN retrieval. | Có kinh nghiệm production: augmentation, PCA, ScaNN/HNSW, Precision@K/QPS. | Chủ yếu image-to-image, chưa khai thác đủ 5 modality. |
| Transformer-Empowered Multi-Modal Item Embedding | MIEM dùng ảnh nhiều góc và product title, kết hợp I2I + multi-modal item embedding tại Shopee. | Tăng semantic awareness và giảm storage so với index từng ảnh. | Phụ thuộc text/title và deployment từng marketplace. |
| MRSE | Multi-modality retrieval system cho large-scale e-commerce search, dùng text, image và user preference. | Tối ưu search công nghiệp với LMoE, hybrid loss, online metrics. | Tập trung nhiều vào text query/user behavior hơn image query thuần. |
| FashionMV | Product-level composed image retrieval với multi-view fashion data. | Nhấn mạnh view incompleteness và product-level representation. | Hẹp miền fashion và yêu cầu dữ liệu multi-view/caption phức tạp. |

## 6.3. Bảng so sánh theo gap

| Method | Sensory Gap | Semantic Gap | Context-Query Gap | Model Gap |
| --- | --- | --- | --- | --- |
| Single image CNN/I2I | Trung bình: học texture/shape tốt nhưng dễ nhạy với crop, compression. | Yếu: dễ trả về sản phẩm nhìn giống nhưng sai category. | Yếu: không hiểu intent ngoài ảnh. | Yếu: kiến thức phụ thuộc category trong dữ liệu train, khó generalize sang nhóm sản phẩm khác. |
| Triplet + attribute + VAE (Shopsy) | Tốt: augmentation và VAE giúp robust hơn. | Trung bình: attribute/triplet giảm sai fine-grained. | Trung bình: tốt cho image query thực tế nhưng ít context modality khác. | Trung bình: vẫn bị giới hạn bởi domain/category của marketplace training data. |
| MIEM | Trung bình-Tốt: nhiều ảnh sản phẩm giảm phụ thuộc một view. | Tốt: title + image giúp hiểu category/semantic. | Trung bình: query vẫn chủ yếu image, context đến từ product title. | Trung bình: phụ thuộc coverage category của image/title trong marketplace đã huấn luyện. |
| MRSE | Trung bình: có image feature nhưng không tối ưu riêng cho noisy image query. | Tốt: text, image, user preference được align. | Tốt: modeling preference theo user/history. | Trung bình: representation đa phương thức vẫn cần dữ liệu đủ đa dạng để bao quát category mới. |
| FashionMV/ProCIR | Tốt: multi-view giảm view incompleteness. | Tốt: product-level embedding và modification text. | Tốt: composed query image + text. | Yếu-Trung bình: chủ yếu được kiểm chứng trong miền fashion. |
| **SCALE + Faiss HNSW/IVF-PQ (ours)** | **Trung bình**: cần kiểm chứng trên ảnh query thực tế. | **Tốt**: SCALE align semantic giữa modality, table/text bổ sung category/attribute. | **Tốt**: ảnh query có thể kèm text/table để làm rõ intent. | **Tốt-Trung bình**: M5Product có 6.232 category giúp mở rộng coverage, nhưng vẫn cần test long-tail/out-of-distribution. |

## 6.4. Vì sao lựa chọn này phù hợp với Product E-commerce Retrieval

Product E-commerce Retrieval có hai yêu cầu khác nhau nhưng phải giải quyết đồng thời. Thứ nhất là **biểu diễn đúng sản phẩm**: hệ thống phải hiểu ảnh, tiêu đề, thuộc tính bảng, video và audio trong cùng một không gian semantic. Thứ hai là **truy hồi nhanh ở quy mô catalog lớn**: sau khi có embedding, hệ thống phải tìm top-K trong hàng trăm nghìn đến hàng triệu sản phẩm mà không duyệt tuyến tính toàn bộ catalog ở mỗi query. Vì vậy, phương pháp được chọn cần tách rõ **feature extraction** và **retrieval indexing**.

### 6.4.1. Vì sao dùng SCALE cho feature extraction?

SCALE phù hợp hơn các hướng image-only vì dữ liệu e-commerce không chỉ nằm trong ảnh. Một ảnh sản phẩm có thể cho biết màu, hình dạng, texture; nhưng title và table lại cho biết brand, material, category, usage scene; video cho biết nhiều góc nhìn; audio hoặc voice có thể mang thông tin mô tả bổ sung. Các nguồn này giúp giảm bốn gap chính:

| Gap | SCALE hỗ trợ như thế nào? |
| --- | --- |
| Sensory Gap | Image/video branch giúp học nhiều góc nhìn; cần đánh giá riêng trên ảnh query nhiễu. |
| Semantic Gap | Text/table branch bổ sung category, brand, material, function để tránh trả về sản phẩm chỉ giống texture nhưng sai ý nghĩa. |
| Context-Query Gap | Ảnh query có thể kèm text/table ngắn để làm rõ intent; nếu không có, hệ thống vẫn truy hồi bằng ảnh. |
| Model Gap | M5Product cung cấp nhiều category và SCALE học từ 5 modality, nên coverage tốt hơn dataset đơn miền. Cần báo cáo metric theo category để nhận ra long-tail hoặc out-of-distribution failure. |

Điểm mạnh quan trọng của SCALE là **Self-harmonized Inter-Modality Contrastive Learning (SIMCL)**. Với 5 modality, không phải cặp modality nào cũng hữu ích như nhau. Ví dụ image-text thường mạnh trong retrieval, image-audio có thể chỉ hữu ích với một số category. SIMCL học trọng số alignment giữa các modality, nhờ đó mô hình không ép mọi modality đóng góp ngang nhau. Điều này phù hợp với catalog e-commerce vì dữ liệu thường thiếu modality, nhiễu và long-tail.

### 6.4.2. Vì sao cần FlatL2/FlatIP baseline?

FlatL2/FlatIP không phải lựa chọn triển khai cuối cho catalog lớn, nhưng nó rất cần trong nghiên cứu vì đây là **exact search baseline**. Nếu HNSW hoặc IVF-PQ trả kết quả kém, ta cần biết lỗi đến từ embedding SCALE hay từ approximation của index.

Flat baseline giúp trả lời ba câu hỏi:

- Embedding của SCALE có đủ phân biệt sản phẩm không?
- ANN index làm mất bao nhiêu recall so với exact search?
- Khi tune HNSW/IVF-PQ, cấu hình nào đạt trade-off tốt nhất giữa speed và accuracy?

Vì vậy, FlatL2/FlatIP là mốc đo chất lượng bắt buộc trước khi kết luận Faiss HNSW hoặc IVF-PQ tốt.

### 6.4.3. Vì sao dùng Faiss HNSW làm index chính?

Faiss HNSW phù hợp làm index chính cho prototype vì:

- Không cần training index như IVF/PQ, nên dễ xây pipeline ban đầu.
- Có recall-latency trade-off tốt thông qua `efSearch`, `efConstruction`, `M`.
- Phù hợp với embedding dense như output của SCALE.
- Dễ so sánh với FlatL2/FlatIP trong cùng thư viện Faiss.
- Phù hợp với yêu cầu demo: cần top-K nhanh, không nhất thiết phải tối ưu memory ngay từ đầu.

Trong Product E-commerce Retrieval, người dùng thường chỉ nhìn top vài kết quả đầu. Vì vậy, index chính cần ưu tiên Precision@K/Recall@K ở K nhỏ và latency thấp. HNSW là lựa chọn cân bằng tốt cho giai đoạn này.

### 6.4.4. Vì sao cần IVF-PQ/OPQ-PQ?

HNSW nhanh và chính xác nhưng tốn RAM vì lưu graph links và vector. Khi catalog tăng lên hàng triệu hoặc hàng chục triệu sản phẩm, memory footprint trở thành bottleneck. IVF-PQ/OPQ-PQ phù hợp làm phương án mở rộng vì:

- IVF chia vector space thành nhiều cụm, lúc search chỉ duyệt một số cụm gần query.
- PQ nén vector thành code ngắn hơn, giảm memory.
- OPQ xoay không gian vector để PQ nén hiệu quả hơn.

Đổi lại, IVF-PQ có thể làm giảm precision/recall và cần training/tuning (`nlist`, `nprobe`, code size). Vì vậy proposal không chọn IVF-PQ làm index đầu tiên, mà dùng nó như phương án scale khi HNSW bắt đầu quá nặng.

### 6.4.5. ScaNN trong related work

Shopsy sử dụng ScaNN như một tham chiếu cho ANN retrieval trong môi trường production. Trong proposal này, ScaNN chỉ được giữ ở related work để đối chiếu định hướng kỹ thuật; nó không thuộc pipeline cài đặt hay kế hoạch benchmark. Phạm vi triển khai được giới hạn ở `IndexFlatIP`, Faiss HNSW và IVF-PQ để mọi index dùng chung một toolkit.

### 6.4.6. Tóm tắt vai trò từng thành phần

| Thành phần | Vai trò trong proposal | Lý do cần có |
| --- | --- | --- |
| SCALE | Multi-modal feature extractor. | Giải quyết semantic gap và tăng tín hiệu biểu diễn từ image, text, table, video, audio. |
| `IndexFlatIP` | Exact baseline. | Đo chất lượng embedding và đo recall loss của approximate index. |
| Faiss HNSW | Index chính. | Dễ triển khai, không cần train, recall-latency tốt cho demo. |
| Faiss IVF-PQ/OPQ-PQ | Index scale/nén. | Giảm memory khi catalog lớn. |
| ScaNN | Tham chiếu related work. | Không thuộc pipeline triển khai của đề tài. |

Như vậy, lựa chọn này phù hợp với Product E-commerce Retrieval vì SCALE tập trung vào chất lượng biểu diễn sản phẩm, `IndexFlatIP` giúp đánh giá khoa học, Faiss HNSW phục vụ demo/latency và IVF-PQ phục vụ scale. ScaNN chỉ là một tham chiếu production trong related work.

## 6.5. Kết luận lựa chọn phương pháp

SCALE phù hợp với feature extraction vì nó được thiết kế cho M5Product, học adaptive modality importance và xử lý missing modality. Với retrieval layer, pipeline triển khai dùng **`IndexFlatIP`** làm exact baseline, **Faiss HNSW** làm index chính vì cân bằng tốt giữa recall và latency, và **Faiss IVF-PQ/OPQ-PQ** khi memory là bottleneck. Các cấu hình này được benchmark bằng Precision@K, Recall@K, QPS, memory footprint và build time.
