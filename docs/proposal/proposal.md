# 00. Introduction

## Topic

**Visual Product Image Search in E-commerce**

## Team Members

| STT | Team Member | Student ID | Email | Phone |
| --- | --- | --- | --- | --- |
| 1 | Trần Hải Đức | 23127173 | thduc23@clc.fitus.edu.vn | 0916821170 |
| 2 | Trần Hoàng Nam | 23127232 | thnam23@clc.fitus.edu.vn | 0916821170 |

## Revision History

| Version | Date | Author | Description |
| --- | --- | --- | --- |
| 0.1 | 2026-06-26 | Trần Hải Đức, Trần Hoàng Nam | Khởi tạo proposal, xác định topic, problem statement, dataset, methodology và plan 2 tháng. |
| 0.2 | 2026-06-26 | Trần Hải Đức, Trần Hoàng Nam | Bổ sung related works, hướng dùng SCALE để trích xuất đặc trưng và Faiss-based ANN retrieval để truy hồi top-K. |

## Scope

Đề tài tập trung xây dựng một hệ thống tìm kiếm sản phẩm trong thương mại điện tử dựa trên truy vấn đa phương thức. Hệ thống nhận query dưới dạng hình ảnh, văn bản, video, audio hoặc bảng thông tin sản phẩm; sau đó dùng SCALE để tạo embedding, dùng FlatL2/FlatIP làm exact baseline, dùng Faiss HNSW làm index chính và trả về danh sách top-K ảnh sản phẩm tương đồng nhất trong catalog.

Proposal này mô tả động lực, đặc trưng dữ liệu product e-commerce image, dataset M5Product, problem statement, related works, methodology dự kiến, tiêu chí đánh giá, kế hoạch thực hiện và danh mục tài liệu tham khảo.

---

# 01. Topic Introduction and Overview

## 1.1. Bối cảnh

E-commerce đang chuyển từ trải nghiệm tìm kiếm dựa hoàn toàn vào từ khóa sang trải nghiệm tìm kiếm giàu ngữ cảnh hơn. Trong tìm kiếm truyền thống, người dùng phải diễn đạt sản phẩm bằng text: tên mặt hàng, màu sắc, kiểu dáng, vật liệu hoặc thương hiệu. Cách này thường thất bại khi người dùng chỉ có một ảnh tham khảo, không biết gọi đúng tên sản phẩm, hoặc khi mô tả của người bán không đồng nhất.

Visual search giải quyết phần lớn ma sát đó bằng cách cho phép người dùng upload ảnh hoặc chụp ảnh sản phẩm ngoài đời, sau đó hệ thống tự nhận diện đặc trưng thị giác và truy hồi sản phẩm tương tự. Bài survey *The Rise of Visual Search in E-Commerce* nhấn mạnh visual search là một công nghệ quan trọng để cải thiện product discovery, tăng relevance, cá nhân hóa và giảm search friction trong hành trình mua sắm.

## 1.2. Visual Product Image Search là gì?

Visual Product Image Search là bài toán tìm kiếm sản phẩm trong catalog dựa trên độ tương đồng giữa query và sản phẩm. Query có thể là:

- Ảnh sản phẩm do người dùng tải lên.
- Text mô tả như "áo khoác denim xanh".
- Video ngắn thể hiện sản phẩm ở nhiều góc nhìn.
- Audio hoặc voice query mô tả nhu cầu.
- Bảng thông tin như brand, material, color, style, usage.

Kết quả không chỉ cần giống về màu hoặc texture, mà còn phải đúng ngữ nghĩa thương mại: cùng loại sản phẩm, cùng style, cùng vật liệu hoặc cùng intent mua hàng. Vì vậy, bài toán này không thể chỉ dựa vào pixel-level similarity.

## 1.3. Tại sao topic này quan trọng?

Visual search hữu ích với e-commerce vì:

- Giảm phụ thuộc vào từ khóa và lỗi mô tả sản phẩm.
- Hỗ trợ mobile commerce và social commerce, nơi người dùng thường thấy sản phẩm qua ảnh.
- Cải thiện discovery cho long-tail products, đặc biệt các sản phẩm khó gọi tên.
- Tăng khả năng matching giữa nhu cầu người mua và catalog người bán.
- Mở đường cho multi-modal retrieval: ảnh + text + bảng thuộc tính + video + audio.

## 1.4. Overview hệ thống đề xuất

Hệ thống được định hướng theo hai tầng chính:

1. **Feature extraction layer**: dùng SCALE trên M5Product để học embedding đa phương thức cho sản phẩm. Mục tiêu là đưa image, text, table, video và audio vào một không gian embedding có thể so sánh.
2. **Retrieval layer**: dùng Faiss-based ANN retrieval để đánh chỉ mục embedding của catalog. Cấu hình chính là Faiss HNSW; FlatL2/FlatIP được dùng làm exact baseline; IVF-PQ được dùng khi cần giảm memory; ScaNN được xem là optional benchmark nếu môi trường hỗ trợ.

Luồng tổng quát:

```mermaid
flowchart LR
    Q["User Query<br/>Image/Text/Video/Audio/Table"] --> P["Preprocess"]
    P --> E["SCALE Feature Extractor"]
    E --> V["Query Embedding"]
    C["Product Catalog"] --> CE["Offline Product Embeddings"]
    CE --> I["Faiss HNSW / IVF-PQ Index"]
    V --> I
    I --> R["Top-K Similar Product Images"]
```

Điểm cốt lõi của đề tài là kết hợp chất lượng embedding đa phương thức với tốc độ truy hồi ở quy mô catalog lớn.

---

# 02. Product E-commerce Image

## 2.1. Khái niệm

Product e-commerce image là ảnh đại diện cho sản phẩm trong môi trường thương mại điện tử. Khác với ảnh tự nhiên thông thường, ảnh sản phẩm thường gắn với mục tiêu mua bán: người dùng cần hiểu sản phẩm là gì, trông như thế nào, có thuộc tính gì, và có phù hợp với nhu cầu hay không.

Trong visual search, ảnh sản phẩm không chỉ là dữ liệu thị giác. Nó là điểm vào để suy luận về category, brand, material, shape, color, pattern, usage scenario và đôi khi cả selling point của sản phẩm.

## 2.2. Các tính chất quan trọng

### 2.2.1. Fine-grained visual similarity

Nhiều sản phẩm khác nhau có hình dáng tổng thể rất giống nhau nhưng khác ở chi tiết nhỏ: logo, pattern, texture, màu phụ, kiểu cổ áo, loại đế giày, model điện thoại. Vì vậy hệ thống cần phân biệt fine-grained attributes thay vì chỉ nhận diện category chung.

### 2.2.2. Semantic similarity

Hai ảnh có thể giống về màu và hình khối nhưng khác ý nghĩa thương mại. Ví dụ hộp pin, hộp nước hoa và hộp bánh có thể đều là hình chữ nhật; nếu chỉ dựa vào texture/shape, hệ thống dễ trả về sản phẩm sai category. Đây là semantic gap: khoảng cách giữa đặc trưng thị giác cấp thấp và ý nghĩa sản phẩm cấp cao.

### 2.2.3. Multi-view and product-level identity

Một sản phẩm thường có nhiều ảnh: mặt trước, mặt sau, cận cảnh, ảnh trên người mẫu, ảnh trong ngữ cảnh sử dụng. Người dùng có thể query bằng một góc nhìn khác với ảnh chính của catalog. Do đó hệ thống cần hiểu product-level identity thay vì chỉ so sánh từng ảnh đơn lẻ.

### 2.2.4. Noisy real-world query

Ảnh query từ người dùng có thể bị crop, nén qua mạng xã hội, xoay, chèn logo, có watermark, có background phức tạp hoặc chứa nhiều object. Bài Shopsy chỉ ra các biến đổi như compression, cropping, scribbling/logo overlay là thách thức thực tế trong visual search cho reseller commerce.

### 2.2.5. Incomplete and heterogeneous metadata

E-commerce catalog thường có thuộc tính không đầy đủ. Sản phẩm này có bảng material/color/brand, sản phẩm khác chỉ có title và ảnh. M5Product cũng phản ánh thực tế này khi có missing modality và long-tail distribution.

## 2.3. Các gap cần xử lý

| Gap | Mô tả | Tác động tới search |
| --- | --- | --- |
| Sensory Gap | Khác biệt giữa ảnh catalog đẹp và ảnh query ngoài đời: ánh sáng, góc chụp, crop, compression. | Làm embedding lệch dù là cùng sản phẩm. |
| Semantic Gap | Đặc trưng pixel giống nhau nhưng category/intent khác nhau. | Trả về sản phẩm nhìn giống nhưng không đúng nhu cầu mua. |
| Context-Query Gap | Query của người dùng thiếu ngữ cảnh hoặc có ngữ cảnh khác catalog. | Khó hiểu người dùng muốn cùng sản phẩm, cùng style hay cùng chức năng. |
| Modal Gap | Query và catalog có modality khác nhau: ảnh query nhưng catalog có text/table/video/audio. | Cần học không gian chung để so sánh cross-modal. |

## 2.4. Quy trình visual search điển hình

Theo survey về visual search trong e-commerce, một hệ thống thường gồm các bước: nhận ảnh query, trích xuất đặc trưng, nhận diện object, so khớp similarity với database và trả về sản phẩm đề xuất. Với đề tài này, quy trình được mở rộng sang multi-modal retrieval: thay vì chỉ dùng CNN trên ảnh, hệ thống dùng multi-modal pretraining để khai thác thêm text, table, video và audio.

```mermaid
flowchart TD
    A["Product Image / User Query"] --> B["Object & Region Understanding"]
    B --> C["Feature Extraction"]
    C --> D["Semantic Embedding"]
    D --> E["Similarity Search"]
    E --> F["Ranked Product Images"]
```

## 2.5. Kết luận chuyển tiếp

Từ các đặc trưng trên, có thể thấy product image search không phải bài toán image retrieval thuần túy. Hệ thống cần vừa chống nhiễu thị giác, vừa hiểu semantic, vừa tận dụng nhiều modality trong catalog. Vì vậy, dataset và phương pháp được chọn phải đủ gần với dữ liệu e-commerce thực tế. Đây là lý do chúng tôi sử dụng M5Product và SCALE làm nền tảng cho phần tiếp theo.

---

# 03. Dataset: M5Product

## 3.1. Chuyển tiếp từ Product E-commerce Image

Ở mục 02, ta thấy dữ liệu e-commerce có ba đặc điểm nổi bật: đa phương thức, nhiễu và thiếu hụt, đồng thời có long-tail category. Nếu chỉ dùng dataset ảnh-text nhỏ hoặc dataset thời trang đơn miền, hệ thống khó mô phỏng môi trường catalog thực tế. Vì vậy, chúng tôi chọn **M5Product** từ bài *M5Product: Self-harmonized Contrastive Learning for E-commercial Multi-modal Pretraining*.

## 3.2. Tổng quan M5Product

M5Product là dataset e-commerce đa phương thức gồm 5 modality:

- Image
- Text/caption
- Table/specification
- Video
- Audio

Theo bài báo, dataset có **6,313,067 samples**, **6,232 categories**, hơn **5,000 attributes**, và lớn hơn đáng kể so với các dataset đa phương thức cùng số modality trước đó. Điểm quan trọng là M5Product không phải dataset paired hoàn hảo; nó chứa missing modality, noise và phân phối long-tail, giống điều kiện e-commerce thật.

## 3.3. Vai trò của từng modality

| Modality | Thông tin chính | Vai trò trong retrieval |
| --- | --- | --- |
| Image | Appearance, color, shape, texture, pattern. | Cốt lõi cho visual similarity và query bằng ảnh. |
| Text | Title, caption, selling point, category phrase. | Bổ sung semantic và intent mà ảnh khó biểu diễn. |
| Table | Brand, material, color, applicable scene, product properties. | Cung cấp thuộc tính cấu trúc, hỗ trợ fine-grained matching. |
| Video | Nhiều góc nhìn, scale, use case, product behavior. | Giảm view incompleteness và sensory gap. |
| Audio | Voice/sound trong video hoặc mô tả liên quan. | Bổ sung tín hiệu khi query hoặc listing có audio. |

## 3.4. Dataset split và annotation

Bài báo M5Product mô tả training set gồm **4,423,160 samples** từ **3,593 classes**. Phần retrieval được chia thành gallery/query cho hai mức:

- **Coarse-grained retrieval**: match theo category, ví dụ tất cả điện thoại được xem là cùng nhóm.
- **Fine-grained retrieval**: chỉ xem là match khi cùng sản phẩm ở mức instance, ví dụ cùng model/màu/kiểu dáng.

Để giảm chi phí labeling, paper dùng ResNet50 và BERT-Base để tạo candidate pool, sau đó dùng crowd-sourcing để xác nhận các cặp match. Điều này phù hợp với mục tiêu của đề tài: đánh giá không chỉ retrieval theo category mà còn retrieval cùng sản phẩm hoặc rất gần sản phẩm.

## 3.5. Vì sao M5Product phù hợp với đề tài?

- Có đủ 5 loại input đúng với problem statement: image, text, video, audio, information table.
- Có dữ liệu lớn để học embedding robust.
- Có missing modality, giúp kiểm tra khả năng vận hành khi catalog không đầy đủ.
- Có category đa dạng hơn các dataset thời trang hẹp miền.
- Có task retrieval, classification và clustering để đánh giá embedding.

## 3.6. Cách sử dụng trong đề tài

Chúng tôi dự kiến dùng M5Product theo ba pha:

1. **Pretraining/finetuning feature extractor**: học embedding chung bằng SCALE.
2. **Build gallery embedding**: trích xuất embedding cho ảnh sản phẩm trong catalog.
3. **Evaluate retrieval**: dùng query set để truy hồi top-K qua FlatL2/FlatIP baseline và Faiss HNSW/IVF-PQ index, sau đó đo mAP@K, Precision@K, Recall@K.

```mermaid
flowchart LR
    D["M5Product<br/>Image Text Table Video Audio"] --> S["Train/Finetune SCALE"]
    S --> G["Gallery Embeddings"]
    S --> Q["Query Embeddings"]
    G --> A["Flat baseline + Faiss HNSW/IVF-PQ"]
    Q --> A
    A --> M["Retrieval Metrics"]
```

---

# 04. Problem Statement

## 4.1. Mục tiêu

Mục tiêu là tìm ra một phương pháp để giải quyết search trên e-commerce: với một catalog gồm nhiều ảnh sản phẩm và một query đa phương thức, hệ thống phải trả về danh sách top-K ảnh sản phẩm giống query nhất.

## 4.2. Formal Definition

| Thành phần | Mô tả |
| --- | --- |
| **Input** | **Dataset D(n)**: Danh sách n hình ảnh Product E-commerce.<br>**Query**: Image, Text, Video, Audio, Information Table. |
| **Output** | Danh sách top-K image giống query nhất. |

Ký hiệu:

- `D = {x_1, x_2, ..., x_n}` là catalog ảnh sản phẩm.
- `q` là query có thể thuộc một hoặc nhiều modality.
- `f(.)` là mô hình trích xuất embedding.
- `sim(f(q), f(x_i))` là độ tương đồng giữa query và ảnh sản phẩm.
- `TopK(q, D)` là K sản phẩm có similarity cao nhất.

Bài toán:

```text
TopK(q, D) = arg top-K_{x_i in D} sim(f(q), f(x_i))
```

## 4.3. Yêu cầu hệ thống

- Trích xuất embedding đủ giàu để biểu diễn visual và semantic similarity.
- Hỗ trợ query đa phương thức.
- Truy hồi nhanh trên catalog lớn.
- Chống nhiễu query như crop, compression, rotation, watermark hoặc background phức tạp.
- Có metric đánh giá rõ ràng cho cả model quality và product/system quality.

## 4.4. Output mong muốn

Với mỗi query, hệ thống trả về:

- Danh sách `top-K` ảnh sản phẩm.
- Similarity score hoặc distance score.
- Product metadata cơ bản để phục vụ UI/ranking sau này.
- Thời gian truy hồi và trạng thái index để phục vụ đánh giá hệ thống.

---

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

---

# 06. Methodology

## 6.1. Mục tiêu kỹ thuật

Phương pháp đề xuất gồm hai khối lớn:

1. **SCALE feature extractor**: biến dữ liệu sản phẩm đa phương thức thành vector embedding chung.
2. **Faiss-based retrieval index**: lưu và tìm kiếm các embedding đó để trả về top-K sản phẩm gần query nhất. Trong đó FlatL2/FlatIP là exact baseline, Faiss HNSW là index chính, Faiss IVF-PQ/OPQ-PQ là phương án khi cần nén, và ScaNN là optional benchmark.

Nói ngắn gọn: SCALE trả lời câu hỏi "query và sản phẩm có giống nhau không?", còn retrieval index trả lời câu hỏi "làm sao tìm nhanh sản phẩm giống nhất trong hàng triệu sản phẩm?".

```mermaid
flowchart LR
    subgraph Offline["Offline Pipeline"]
        D["M5Product / Catalog"] --> PP["Preprocess by Modality"]
        PP --> FE["SCALE Feature Extractor"]
        FE --> PE["Product Embeddings"]
        PE --> IDX["Build Faiss HNSW / IVF-PQ Index"]
    end
    subgraph Online["Online Query Pipeline"]
        Q["Query<br/>Image/Text/Video/Audio/Table"] --> QP["Preprocess"]
        QP --> QF["SCALE Query Encoder"]
        QF --> QE["Query Embedding"]
        QE --> IDX
        IDX --> TOP["Top-K Product Images"]
    end
```

## 6.2. Một khái niệm nền: token, feature và embedding

Trước khi đi vào từng modality, cần phân biệt ba khái niệm:

- **Raw data**: dữ liệu gốc, ví dụ ảnh `.jpg`, câu mô tả sản phẩm, bảng key-value, video `.mp4`, audio waveform.
- **Feature/token**: biểu diễn trung gian mà model đọc được. Transformer không đọc trực tiếp ảnh/video/audio thô; ta phải đổi chúng thành một chuỗi vector. Mỗi vector trong chuỗi được gọi là một token.
- **Embedding**: vector cuối cùng đại diện cho cả query hoặc cả sản phẩm. Vector này được dùng để tính similarity và đưa vào Flat baseline hoặc Faiss HNSW/IVF-PQ index.

Ví dụ với text `"white leather sneakers"`, tokenizer có thể tách thành các token như `[CLS]`, `white`, `leather`, `sneakers`, `[SEP]`; mỗi token được biến thành một vector 768 chiều. Với image, "token" không phải là word mà là region feature: vùng giày, vùng logo, vùng đế giày, vùng texture, v.v.

## 6.3. Tổng quan kiến trúc SCALE

SCALE trong paper M5Product là một kiến trúc multi-modal pretraining gồm:

1. **Modality-specific encoders**: mỗi loại dữ liệu có encoder riêng để biến dữ liệu đó thành token vectors.
2. **Concatenate tokens**: nối token của các modality thành một sequence dài.
3. **Joint Co-Transformer (JCT)**: transformer chung học quan hệ giữa token của nhiều modality.
4. **SIMCL + masked tasks**: các loss để model học alignment giữa modality và học lại phần bị che.

```mermaid
flowchart TD
    I["Image Regions<br/>Faster R-CNN + ResNet101"] --> IE["Image Transformer"]
    T["Text Tokens<br/>BERT tokenizer/init"] --> TE["Text Transformer"]
    TB["Table Entities<br/>key-value attributes"] --> TBE["Table Transformer"]
    V["Video Frames<br/>sampled frame tokens"] --> VE["Video Transformer"]
    A["Audio MFCC<br/>spectrogram-like tokens"] --> AE["Audio Transformer"]
    IE --> CAT["Concatenate Tokens<br/>+ modality embeddings + masks"]
    TE --> CAT
    TBE --> CAT
    VE --> CAT
    AE --> CAT
    CAT --> JCT["Joint Co-Transformer"]
    JCT --> Z["Unified Product Embedding"]
```

Theo paper M5Product/SCALE:

- Text transformer được khởi tạo từ BERT.
- Các modality transformer và fusion encoder có 6 transformer layers; tổng single-modality + multi-modal fusion là 12 layers.
- Hidden size là 768.
- Caption length tối đa là 36, table length tối đa là 64.
- Image dùng Faster R-CNN với ResNet101 pretrained trên Visual Genome để lấy 10-36 region features.
- Audio dùng MFCC.
- Missing modality được xử lý bằng zero imputation.

## 6.4. Image branch: Image Regions -> Image Transformer

### 6.4.1. Image branch là gì?

Image branch là nhánh biến ảnh sản phẩm thành một sequence các vector mô tả những vùng quan trọng trong ảnh. Thay vì đưa toàn bộ ảnh vào transformer như một ma trận pixel, SCALE dùng object/region features.

Ví dụ ảnh một đôi giày:

- Region 1: toàn bộ đôi giày.
- Region 2: logo.
- Region 3: phần đế.
- Region 4: dây giày.
- Region 5: texture da/vải.

Mỗi region được biểu diễn bằng một vector. Sequence các vector này là input cho Image Transformer.

### 6.4.2. Vì sao dùng region feature thay vì toàn ảnh?

Trong e-commerce, background, model, ánh sáng và layout ảnh có thể thay đổi mạnh. Nếu dùng toàn ảnh, model dễ học nhầm background hoặc style chụp. Region feature giúp model tập trung vào object chính và các bộ phận sản phẩm.

Paper SCALE dùng hướng **bottom-up attention**: object detector đề xuất các vùng ảnh quan trọng trước, sau đó transformer học quan hệ giữa các vùng đó.

### 6.4.3. Input và output

| Thành phần | Mô tả |
| --- | --- |
| Input raw | Ảnh sản phẩm hoặc ảnh query. |
| Detector output | `N` bounding boxes, thường chọn 10-36 vùng có objectness score cao. |
| Region feature | Vector cho mỗi vùng, ví dụ `N x d`. |
| Image transformer output | Sequence token ảnh đã được contextualize, ví dụ `N x 768`. |

### 6.4.4. Tool đề xuất

| Tool | Nguồn | Vai trò |
| --- | --- | --- |
| `airsplay/py-bottom-up-attention` | Được paper SCALE nhắc tới trong footnote. | Gần nhất với cách SCALE lấy bottom-up region features. |
| `torchvision.models.detection` | Tool ngoài, dựa trên PyTorch/TorchVision. | Dùng Faster R-CNN/ResNet-FPN để prototype object detection nếu không dùng đúng bottom-up attention code. |
| `timm` hoặc `torchvision.models` | Tool ngoài. | Dùng ResNet/ViT/Swin làm visual backbone thay thế nếu cần đơn giản hóa. |

Nếu dùng đúng tinh thần paper, lựa chọn tốt nhất là `py-bottom-up-attention`. Nếu môi trường khó cài, có thể dùng `torchvision` Faster R-CNN để lấy bounding boxes, sau đó lấy feature từ backbone hoặc ROI pooled features. Đây là thay thế thực dụng, cần ghi rõ trong báo cáo là implementation approximation.

### 6.4.5. Thiết kế thêm của nhóm

Paper nói image region extraction nhưng không mô tả chi tiết augmentation cho query thực tế. Chúng tôi bổ sung augmentation dựa trên Shopsy:

- JPEG compression.
- Random crop.
- Rotation nhẹ.
- Horizontal flip.
- Logo/watermark overlay.
- Resize quality degradation.

Mục tiêu là giảm **sensory gap** giữa ảnh catalog đẹp và ảnh query ngoài đời.

## 6.5. Text branch: Text Tokens -> Text Transformer

### 6.5.1. Text branch là gì?

Text branch biến title/caption/description thành token vectors. Ví dụ:

```text
Input text: "Bubble Matt Blind Box Storage Ladder"
Tokens: [CLS], bubble, matt, blind, box, storage, ladder, [SEP]
```

Mỗi token được ánh xạ thành vector thông qua embedding layer của BERT, sau đó đi qua Text Transformer để học ngữ cảnh.

### 6.5.2. BERT init nghĩa là gì?

SCALE không train text transformer từ con số 0. Paper dùng BERT để khởi tạo text transformer. BERT là encoder-only transformer đã học ngôn ngữ bằng masked language modeling. Vì vậy, ngay từ đầu model đã biết một phần quan hệ giữa các từ, ví dụ `leather`, `shoe`, `sneaker`, `white`, `size`.

### 6.5.3. Input và output

| Thành phần | Mô tả |
| --- | --- |
| Input raw | Product title, caption, description. |
| Tokenizer output | `input_ids`, `attention_mask`, optional `token_type_ids`. |
| Text transformer output | Sequence token text, ví dụ `L_text x 768`. |
| Vai trò | Bổ sung category, function, material, selling point mà ảnh không thể hiện rõ. |

### 6.5.4. Tool đề xuất

| Tool | Nguồn | Vai trò |
| --- | --- | --- |
| Hugging Face `transformers` | Tool ngoài, official docs. | Dùng `BertTokenizer`, `BertModel`, hoặc `AutoTokenizer/AutoModel`. |
| Google BERT checkpoint | Paper gốc BERT. | Khởi tạo text encoder. |

Ví dụ lựa chọn implementation:

```text
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_encoder = AutoModel.from_pretrained("bert-base-uncased")
```

Nếu dữ liệu có tiếng Việt hoặc đa ngôn ngữ, có thể đổi sang `bert-base-multilingual-cased` hoặc PhoBERT. Đây là thiết kế thêm của nhóm nếu dataset/query tiếng Việt xuất hiện trong demo.

## 6.6. Table branch: Table Entities -> Table Transformer

### 6.6.1. Table entities là gì?

Table trong e-commerce là thông tin có cấu trúc dạng key-value. Ví dụ:

| Key | Value |
| --- | --- |
| Item | Blind Box Ladder Storage Box |
| Brand | Tang Craftsman |
| Material | Wood |
| Color | White, Light Gray, Dark Gray |
| Applicable Scene | Study |

Một **table entity** là một đơn vị thuộc tính có nghĩa, thường là cặp `key: value`. Ví dụ `Material: Wood` là một entity, `Color: White` là một entity. Nó khác text bình thường vì key cho biết vai trò của value.

### 6.6.2. Vì sao không chỉ nối table thành text?

Nếu nối mọi thứ thành câu text, model có thể mất cấu trúc key-value. Ví dụ `white` trong `Color: White` khác với `Brand: White Label`. Table Transformer giúp model học rằng `Color`, `Brand`, `Material`, `Size`, `Applicable Scene` là các loại thuộc tính khác nhau.

### 6.6.3. Cách biểu diễn table entity

Paper SCALE nói table encoder là transformer riêng và dùng **Mask Entity Modeling (MEM)**. Paper không đưa đầy đủ code encoding table entity trong phần chính, nên chúng tôi đề xuất một serialization rõ ràng như sau:

```text
[ENT] key = material [VAL] wood [SEP]
[ENT] key = color [VAL] white [SEP]
[ENT] key = brand [VAL] tang craftsman [SEP]
```

Đây là **thiết kế thêm của nhóm**, không phải tool có sẵn từ paper. Lý do thiết kế:

- Giữ được ranh giới từng entity.
- Giữ được phân biệt key và value.
- Cho phép mask nguyên entity trong MEM.
- Dễ implement bằng tokenizer BERT hoặc tokenizer riêng.

### 6.6.4. Input và output

| Thành phần | Mô tả |
| --- | --- |
| Input raw | JSON/CSV/key-value product specification. |
| Entity sequence | Danh sách entity đã serialize. |
| Table transformer output | Sequence token/entity table, ví dụ `L_table x 768`. |
| Vai trò | Bổ sung fine-grained attributes như brand, material, color, size. |

### 6.6.5. Tool đề xuất

| Tool | Nguồn | Vai trò |
| --- | --- | --- |
| `pandas` | Tool ngoài. | Đọc CSV/JSON, normalize bảng thuộc tính. |
| Python `json` | Standard library. | Parse product attribute JSON. |
| Hugging Face tokenizer | Tool ngoài. | Tokenize chuỗi entity serialization. |

### 6.6.6. Mask Entity Modeling

Với MLM, ta mask token lẻ. Với MEM, ta mask cả entity:

```text
Before: [ENT] key = material [VAL] wood [SEP]
After:  [MASK_ENTITY]
Target: material = wood
```

Việc mask nguyên entity buộc model dùng image/text/video/audio còn lại để suy luận thuộc tính bị thiếu. Ví dụ nhìn ảnh ghế gỗ và title "wooden chair", model có thể dự đoán `Material: Wood`.

## 6.7. Video branch: Video Frames -> Video Transformer

### 6.7.1. Video branch là gì?

Video branch biến video sản phẩm thành chuỗi frame features. Một video có nhiều frame, nhưng không thể đưa toàn bộ frame vào model vì quá nặng. Ta sample một số frame đại diện, ví dụ 8 hoặc 16 frame theo thời gian.

Ví dụ video quay túi xách:

- Frame 1: mặt trước.
- Frame 2: góc nghiêng.
- Frame 3: mặt sau.
- Frame 4: cận cảnh khóa kéo.
- Frame 5: bên trong túi.

Những frame này giúp model hiểu sản phẩm ở nhiều góc nhìn.

### 6.7.2. Input và output

| Thành phần | Mô tả |
| --- | --- |
| Input raw | Video sản phẩm. |
| Frame sampler | Chọn `T` frame theo thời gian. |
| Frame feature | Vector cho từng frame hoặc region trong frame. |
| Video transformer output | Sequence video tokens, ví dụ `T x 768`. |

### 6.7.3. Tool đề xuất

| Tool | Nguồn | Vai trò |
| --- | --- | --- |
| `ffmpeg` | Tool ngoài, industry standard. | Decode video, extract frames/audio. |
| `PyAV` | Tool ngoài, Python binding cho FFmpeg libraries. | Đọc video frame trực tiếp trong Python dataloader. |
| `decord` | Tool ngoài. | Efficient video loading cho deep learning. |
| `torchvision.io` | Tool ngoài. | Đọc video cơ bản trong PyTorch ecosystem. |

### 6.7.4. Thiết kế thêm của nhóm

Paper nói "ordinal frames sampled from video are fed into video encoder" nhưng không chốt sampling policy. Chúng tôi đề xuất:

- **Uniform sampling**: lấy `T` frame cách đều toàn video, đơn giản và ổn định.
- **Middle-biased sampling**: lấy nhiều frame ở giữa video nếu video đầu/cuối có intro/outro.
- **Object-aware sampling**: nếu có detector, ưu tiên frame có object confidence cao.

Giai đoạn đầu nên dùng uniform sampling vì dễ tái lập. Object-aware sampling là hướng nâng cấp nếu video noisy.

## 6.8. Audio branch: Audio MFCC -> Audio Transformer

### 6.8.1. Audio branch là gì?

Audio branch biến tín hiệu âm thanh thành chuỗi feature theo thời gian. Trong SCALE, audio được biểu diễn bằng **MFCC - Mel-Frequency Cepstral Coefficients**.

MFCC là cách nén phổ âm thanh theo thang Mel, gần với cách tai người cảm nhận tần số. Nó thường gồm các bước:

1. Chia audio thành frame ngắn.
2. Tính phổ tần số cho từng frame.
3. Áp dụng Mel filter bank.
4. Lấy log năng lượng.
5. Dùng DCT để tạo cepstral coefficients.

### 6.8.2. Audio giúp gì cho product search?

Audio không phải modality mạnh nhất cho mọi sản phẩm, nhưng có thể hữu ích khi:

- Video sản phẩm có lời giới thiệu.
- Query là voice/audio.
- Sản phẩm có âm thanh đặc trưng, ví dụ nhạc cụ, thiết bị điện, đồ chơi.
- Audio transcript có thể bổ sung text signal.

### 6.8.3. Input và output

| Thành phần | Mô tả |
| --- | --- |
| Input raw | Audio waveform hoặc audio track từ video. |
| MFCC output | Ma trận `time_steps x n_mfcc`. |
| Projection | Linear layer đưa MFCC về hidden size 768. |
| Audio transformer output | Sequence audio tokens, ví dụ `L_audio x 768`. |

### 6.8.4. Tool đề xuất

| Tool | Nguồn | Vai trò |
| --- | --- | --- |
| `librosa.feature.mfcc` | Tool ngoài, official docs. | Tính MFCC từ waveform. |
| `torchaudio.transforms.MFCC` | Tool ngoài, PyTorch ecosystem. | Tính MFCC trực tiếp bằng tensor pipeline. |
| `ffmpeg` hoặc `PyAV` | Tool ngoài. | Tách audio track từ video. |

Nếu audio là voice query, nhóm có thể thêm ASR để chuyển speech sang text rồi đưa vào Text Transformer. Đây là hướng mở rộng, không phải thành phần bắt buộc của SCALE gốc.

## 6.9. Concatenate Tokens: nối các modality như thế nào?

Sau khi từng encoder tạo token sequence riêng, ta nối chúng thành một sequence chung:

```text
[IMG_CLS], img_1, img_2, ..., img_N,
[TXT_CLS], txt_1, txt_2, ..., txt_L,
[TAB_CLS], tab_1, tab_2, ..., tab_M,
[VID_CLS], vid_1, vid_2, ..., vid_T,
[AUD_CLS], aud_1, aud_2, ..., aud_A
```

Để JCT biết token nào thuộc modality nào, cần cộng thêm:

- **Position embedding**: token đứng ở vị trí nào trong sequence.
- **Modality/type embedding**: token thuộc image, text, table, video hay audio.
- **Attention mask**: token nào là thật, token nào là padding/missing.

Đây là phần implementation cần làm rõ khi code. Nếu thiếu modality, ví dụ query chỉ có image, ta dùng mask để JCT không chú ý vào padding token của modality khác.

## 6.10. Joint Co-Transformer (JCT)

### 6.10.1. JCT là gì?

JCT là transformer chung nhận sequence token đã nối từ nhiều modality. Nó dùng self-attention để mỗi token có thể "nhìn" các token khác, kể cả token từ modality khác.

Ví dụ:

- Token ảnh vùng "đế giày" có thể chú ý tới token text "sneaker".
- Token table `Material: leather` có thể chú ý tới vùng texture trong ảnh.
- Token video frame cận cảnh logo có thể chú ý tới text brand.
- Token audio từ lời giới thiệu có thể chú ý tới table attribute.

### 6.10.2. Vì sao JCT quan trọng?

Nếu chỉ encode từng modality riêng rồi average, model khó học quan hệ chi tiết giữa chúng. JCT cho phép cross-modal reasoning:

- Text giải thích ảnh.
- Table xác nhận thuộc tính trong ảnh.
- Video bổ sung góc nhìn ảnh không có.
- Audio/voice bổ sung intent hoặc selling point.

JCT là nơi semantic gap và modal gap được giảm mạnh nhất.

### 6.10.3. Self-attention trong JCT hoạt động thế nào?

Với mỗi token, transformer tạo ba vector `Q`, `K`, `V`:

- `Q` - query: token này đang tìm thông tin gì?
- `K` - key: token khác chứa loại thông tin nào?
- `V` - value: nội dung token khác sẽ truyền sang là gì?

Attention score giữa token `i` và token `j` được tính từ `Q_i` và `K_j`. Nếu score cao, token `i` lấy nhiều thông tin từ `V_j`.

Trong JCT, token image có thể attend tới token text/table/video/audio. Vì vậy, output của JCT không còn là feature đơn modality nữa mà là feature đã được "làm giàu" bởi các modality khác.

```mermaid
flowchart TD
    X["Concatenated Multi-modal Tokens"] --> SA["Multi-head Self-Attention"]
    SA --> FFN["Feed Forward Network"]
    FFN --> LN["Residual + LayerNorm"]
    LN --> O["Contextualized Multi-modal Tokens"]
    O --> POOL["Pooling / CLS Selection"]
    POOL --> EMB["Unified Embedding"]
```

### 6.10.4. Output của JCT lấy embedding như thế nào?

Có ba cách phổ biến:

| Cách pooling | Mô tả | Ghi chú |
| --- | --- | --- |
| Global `[CLS]` token | Thêm một token đại diện toàn sample, lấy output token này. | Dễ dùng cho retrieval. |
| Mean pooling | Trung bình các token hợp lệ sau JCT. | Ổn nếu sequence không có CLS tốt. |
| Modality-aware pooling | Pool từng modality rồi học trọng số fusion. | Phù hợp nếu muốn giải thích modality contribution. |

Paper nói dùng fused modality features từ JCT. Trong implementation của nhóm, đề xuất dùng global `[CLS]` hoặc mean pooling ở bản đầu, sau đó thử modality-aware pooling nếu cần explainability.

## 6.11. Self-supervised masked tasks

SCALE dùng các pretext tasks để model học đặc trưng hữu ích ngay cả khi không có label thủ công.

| Task | Modality | Cách hoạt động | Vì sao hữu ích |
| --- | --- | --- | --- |
| MRP - Masked Region Prediction | Image | Che một số region image và dự đoán lại feature/label vùng đó. | Học object part và visual context. |
| MLM - Masked Language Modeling | Text | Che token text và dự đoán token bị che. | Học ngữ nghĩa title/caption. |
| MEM - Mask Entity Modeling | Table | Che nguyên entity key-value. | Học product attributes có cấu trúc. |
| MFP - Mask Frame Prediction | Video | Che frame/token video và dự đoán lại. | Học quan hệ theo thời gian/góc nhìn. |
| MAM - Mask Audio Modeling | Audio | Che audio feature và dự đoán lại. | Học pattern âm thanh/speech context. |

Paper mask 15% input. Với table, mask nguyên entity giúp model học tốt hơn so với mask từng word rời rạc.

## 6.12. Self-harmonized Inter-Modality Contrastive Learning (SIMCL)

Nếu chỉ có hai modality image-text, ta có thể dùng contrastive learning: image và text của cùng sản phẩm là positive pair, image và text của sản phẩm khác là negative pair. Nhưng với 5 modality, có nhiều cặp: image-text, image-table, image-video, text-table, video-audio, v.v. Không phải cặp nào cũng quan trọng như nhau.

SIMCL học một **modality alignment score matrix** để tự cân bằng:

- Cặp modality nào align tốt và nhiều thông tin hơn thì trọng số cao hơn.
- Cặp modality nhiễu hoặc thiếu thông tin thì trọng số thấp hơn.
- Masked task của từng modality cũng được cân bằng, tránh một modality lấn át toàn bộ training.

```mermaid
flowchart TD
    B["Mini-batch with M Modalities"] --> P["Positive pairs<br/>same product"]
    B --> N["Negative pairs<br/>different products"]
    P --> CL["Inter-Modality Contrastive Loss"]
    N --> CL
    MASK["Masked Tasks<br/>MRP MLM MEM MFP MAM"] --> ML["Intra-Modality Masked Loss"]
    S["Learned Alignment Score Matrix"] --> W1["Weight pair losses"]
    S --> W2["Weight modality losses"]
    CL --> W1
    ML --> W2
    W1 --> L["Total SCALE Loss"]
    W2 --> L
```

Tổng loss khái niệm:

```text
L_total = weighted_inter_modality_contrastive_loss
        + weighted_intra_modality_masked_loss
```

Đây là lý do SCALE phù hợp hơn cách fusion đơn giản như concatenate rồi train classifier: nó không chỉ nối dữ liệu, mà còn học mức độ tin cậy/đóng góp giữa các modality.

## 6.13. Embedding extraction

Sau khi train/finetune, ta dùng SCALE để tạo embedding:

### Catalog embedding offline

1. Load product record.
2. Đọc image/text/table/video/audio nếu có.
3. Preprocess từng modality.
4. Chạy qua modality encoders.
5. Nối token và chạy JCT.
6. Pool output thành vector embedding.
7. L2-normalize.
8. Lưu `{product_id, image_id, embedding, metadata}`.

### Query embedding online

1. Nhận query.
2. Xác định query có modality nào.
3. Preprocess modality đó.
4. Các modality thiếu được mask/zero.
5. Chạy qua SCALE.
6. L2-normalize embedding.
7. Search Flat baseline hoặc Faiss HNSW/IVF-PQ index.

L2-normalization giúp cosine similarity hoặc inner product ổn định hơn. Shopsy cũng dùng L2-normalization trước khi đưa embedding vào ANN.

## 6.14. Retrieval index: Flat baseline, Faiss HNSW, IVF-PQ và ScaNN

Exhaustive search phải so sánh query với mọi embedding trong catalog, không phù hợp với hàng triệu sản phẩm. Vì vậy retrieval layer dùng Approximate Nearest Neighbor, nhưng proposal cần chọn index cụ thể thay vì chỉ ghi chung chung là ANN.

Chiến lược thống nhất của đề tài:

- **FlatL2/FlatIP**: exact baseline để biết chất lượng tối đa khi không approximate.
- **Faiss HNSW**: index chính cho prototype/demo vì không cần training, recall-latency tốt và dễ benchmark.
- **Faiss IVF-PQ hoặc OPQ-PQ**: phương án scale khi memory là bottleneck.
- **ScaNN**: optional benchmark nếu môi trường cài đặt hỗ trợ; không phụ thuộc vào ScaNN để hoàn thành demo.
- **Qdrant/Milvus**: chỉ dùng nếu cần product-level service có metadata filtering/API sẵn, không phải trọng tâm thuật toán trong proposal.

| Index | Ưu điểm | Nhược điểm | Khi dùng |
| --- | --- | --- | --- |
| FlatL2 / FlatIP | Chính xác, baseline tốt. | QPS thấp khi catalog lớn. | Đánh giá oracle/baseline. |
| Faiss HNSW | Recall-latency tốt, không cần training, dễ prototype. | Tốn RAM, cần tune `M`, `efSearch`, `efConstruction`; không tối ưu khi cần xóa vector thường xuyên. | Index chính cho demo và benchmark ban đầu. |
| Faiss IVF-PQ / OPQ-PQ | Giảm memory, có thể scale lớn hơn HNSW. | Cần training index, có thể giảm precision, cần tune `nlist`, `nprobe`, PQ code size. | Khi memory là bottleneck hoặc catalog tăng lớn. |
| ScaNN | Tối ưu vector similarity search bằng pruning/quantization, QPS CPU cao trong một số setting. | Phụ thuộc platform và tuning, không cần là dependency bắt buộc. | Optional benchmark dựa trên kinh nghiệm Shopsy. |
| Annoy | Đơn giản, dễ dùng. | Build/search có thể kém hơn ScaNN/HNSW trong setting lớn. | Prototype nhanh. |

Từ Shopsy, ScaNN và HNSW đạt Precision@4 tương đương FlatL2 nhưng QPS cao hơn nhiều trên index 3 triệu ảnh. Vì vậy đề tài sẽ ưu tiên:

1. FlatL2/FlatIP làm exact baseline.
2. Faiss HNSW làm default retrieval index.
3. Faiss IVF-PQ/OPQ-PQ nếu cần giảm memory.
4. ScaNN nếu môi trường hỗ trợ tốt và còn thời gian benchmark.

## 6.15. Index building pipeline

```mermaid
flowchart TD
    C["Catalog Products"] --> E["SCALE Embedding Extraction"]
    E --> N["L2 Normalize"]
    N --> P["Optional PCA<br/>only if memory/latency requires it"]
    P --> B["Build Faiss HNSW / IVF-PQ Index"]
    B --> V["Validate Recall/Precision vs FlatL2/FlatIP"]
    V --> S["Save Index + Metadata Mapping"]
```

Nếu embedding dimension lớn và index memory cao, có thể áp dụng PCA giống Shopsy. Với SCALE hidden size 768, PCA cần được kiểm tra thực nghiệm: chỉ giữ nếu giảm memory/latency mà không làm mất nhiều Precision@K.

## 6.16. Training workflow

```mermaid
flowchart TD
    D["M5Product Training Split"] --> A["Data Loader<br/>5 modalities + missing masks"]
    A --> AUG["Image/Query Augmentation"]
    AUG --> ENC["Modality Encoders"]
    ENC --> CAT["Concatenate Tokens"]
    CAT --> JCT["Joint Co-Transformer"]
    JCT --> MT["Masked Tasks"]
    JCT --> IMCL["Inter-Modality Contrastive Learning"]
    MT --> SIMCL["SIMCL Weighting"]
    IMCL --> SIMCL
    SIMCL --> LOSS["Total Loss"]
    LOSS --> OPT["Adam Optimizer"]
    OPT --> CKPT["Model Checkpoint"]
```

Training chia thành:

- **Pretraining**: self-supervised masked tasks + SIMCL trên dữ liệu lớn.
- **Finetuning**: dùng subset retrieval/classification nếu có label category/instance.
- **Embedding export**: freeze model tốt nhất và export gallery/query embedding.

## 6.17. Testing and evaluation workflow

```mermaid
flowchart TD
    Q["Query Set"] --> QE["Query Embeddings"]
    G["Gallery Set"] --> GE["Gallery Embeddings"]
    GE --> IDX["Faiss HNSW / IVF-PQ Index"]
    QE --> IDX
    IDX --> TOPK["Top-K Results"]
    TOPK --> GT["Compare with Ground Truth"]
    GT --> M["mAP@K / Precision@K / Recall@K / NDCG@K"]
    IDX --> SYS["Latency / QPS / Memory / Build Time"]
```

Model quality được đo bằng retrieval metrics. System quality được đo bằng latency, QPS, memory footprint và index build/update time.

## 6.18. Serving design

Online serving gồm:

1. Query API nhận file hoặc payload.
2. Preprocessor chuẩn hóa modality.
3. SCALE inference tạo query embedding.
4. ANN service trả về candidate IDs.
5. Metadata service map IDs sang image/product.
6. Optional reranker sắp xếp lại bằng business rules hoặc score kết hợp.

```mermaid
sequenceDiagram
    participant U as User
    participant API as Search API
    participant FE as SCALE Encoder
    participant ANN as Faiss Index
    participant DB as Product DB
    U->>API: Submit query
    API->>FE: Preprocessed modality payload
    FE-->>API: Query embedding
    API->>ANN: Search top-K
    ANN-->>API: Product/Image IDs + distances
    API->>DB: Fetch metadata/images
    DB-->>API: Product records
    API-->>U: Ranked product images
```

## 6.19. Rủi ro và hướng xử lý

| Rủi ro | Nguyên nhân | Hướng xử lý |
| --- | --- | --- |
| Query chỉ có image nhưng model train nhiều modality | Modal mismatch giữa train và serve. | Missing modality mask/zero imputation, train với random modality dropout. |
| Semantic sai dù ảnh giống | Embedding quá thiên về texture. | Tăng trọng số text/table, finetune bằng category/instance labels. |
| Approximate index giảm recall | Approximation/quantization quá mạnh. | So sánh với FlatL2/FlatIP, tune Faiss HNSW hoặc IVF-PQ, dùng rerank top-N bằng exact distance. |
| Latency cao | SCALE inference nặng. | Cache embedding, batch offline, dùng model distillation/ONNX/TensorRT nếu cần. |
| Catalog update | Product mới cần embedding và index update. | Incremental index hoặc rebuild định kỳ theo batch. |

## 6.20. Summary

SCALE + Faiss-based retrieval phù hợp với đề tài vì SCALE học representation chung cho 5 modality và tự cân bằng đóng góp giữa các modality, còn Flat/HNSW/IVF-PQ biến embedding đó thành hệ thống truy hồi có thể đo đạc và mở rộng. Điểm cần nhấn mạnh là mỗi block trong kiến trúc đều có vai trò cụ thể: image branch học vùng sản phẩm, text branch học mô tả, table branch học thuộc tính có cấu trúc, video branch học góc nhìn theo thời gian, audio branch học tín hiệu âm thanh/voice, JCT học quan hệ giữa tất cả token, SIMCL học cách cân bằng modality, và Faiss HNSW/IVF-PQ phục vụ truy hồi top-K ở quy mô lớn.

---

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

---

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

---

# 09. Appendix and References

## 9.1. References từ local papers

1. N. Venkatesan, M. Suresh, Vethamuthu Richard Paul, Diganta Kumar Das, P. Vijayakumar. **The Rise of Visual Search in E-Commerce: Leveraging AI to Redefine Product Discovery**. Journal of Marketing & Social Research, 2025.
2. Xiao Dong, Xunlin Zhan, Yangxin Wu, Yunchao Wei, Michael C. Kampffmeyer, Xiao-Yong Wei, Minlong Lu, Yaowei Wang, Xiaodan Liang. **M5Product: Self-harmonized Contrastive Learning for E-commercial Multi-modal Pretraining**. 2022.
3. Prajit Nadkarni, Narendra Varma Dasararaju. **Visually Similar Products Retrieval for Shopsy**. arXiv:2210.04560, 2022.
4. Chang Liu, Peng Hou, Anxiang Zeng, Han Yu. **Transformer-Empowered Multi-Modal Item Embedding for Enhanced Image Search in E-commerce**. AAAI, 2024.
5. Hao Jiang, Haoxiang Zhang, Qingshan Hou, Chaofeng Chen, Weisi Lin, Jingchang Zhang, Annan Wang. **MRSE: An Efficient Multi-modality Retrieval System for Large Scale E-commerce**.
6. Peng Yuan, Bingyin Mei, Hui Zhang. **FashionMV: Product-Level Composed Image Retrieval with Multi-View Fashion Data**. arXiv:2604.10297, 2026.

## 9.2. References được trích trong methodology

7. Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**. IEEE TPAMI, 2017.
8. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. **Deep Residual Learning for Image Recognition**. CVPR, 2016.
9. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. NAACL, 2019.
10. Alec Radford et al. **Learning Transferable Visual Models From Natural Language Supervision**. ICML, 2021.
11. Florian Schroff, Dmitry Kalenichenko, James Philbin. **FaceNet: A Unified Embedding for Face Recognition and Clustering**. CVPR, 2015.
12. Diederik P. Kingma, Max Welling. **Auto-Encoding Variational Bayes**. ICLR, 2014.
13. Yury A. Malkov, Dmitry A. Yashunin. **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs**. IEEE TPAMI, 2020.
14. Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, Sanjiv Kumar. **Accelerating Large-Scale Inference with Anisotropic Vector Quantization**. ICML, 2020.
15. Jeff Johnson, Matthijs Douze, Herve Jegou. **Billion-scale Similarity Search with GPUs**. IEEE Transactions on Big Data, 2019.
16. Herve Jegou, Matthijs Douze, Cordelia Schmid. **Product Quantization for Nearest Neighbor Search**. IEEE TPAMI, 2011.

## 9.3. Tool/library references

17. Meta AI. **Faiss Documentation**. https://faiss.ai/
18. Meta AI. **Faiss Guidelines to Choose an Index**. https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
19. Google Research. **ScaNN: Scalable Nearest Neighbors**. https://github.com/google-research/google-research/tree/master/scann
20. Qdrant. **Indexing Documentation**. https://qdrant.tech/documentation/manage-data/indexing/
21. Milvus. **In-memory Index Documentation**. https://milvus.io/docs/index.md
22. NVIDIA. **Apex: Tools for Easy Mixed Precision and Distributed Training in PyTorch**. https://github.com/NVIDIA/apex
23. PyTorch Contributors. **PyTorch**. https://pytorch.org/
24. Hugging Face. **BERT model documentation in Transformers**. https://huggingface.co/docs/transformers/model_doc/bert
25. Hugging Face. **Transformers library**. https://github.com/huggingface/transformers
26. TorchVision. **Detection models and operators**. https://pytorch.org/vision/stable/models.html
27. Librosa. **librosa.feature.mfcc documentation**. https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
28. PyAV Contributors. **PyAV Documentation**. https://pyav.org/docs/stable/
29. FFmpeg Developers. **FFmpeg Documentation**. https://ffmpeg.org/documentation.html
30. Pandas Contributors. **pandas Documentation**. https://pandas.pydata.org/docs/
31. Ross Wightman. **timm: PyTorch Image Models**. https://github.com/huggingface/pytorch-image-models

## 9.4. Paper references for tools and model components

32. Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang. **Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering**. CVPR, 2018.
33. Ranjay Krishna et al. **Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations**. International Journal of Computer Vision, 2017.
34. Ashish Vaswani et al. **Attention Is All You Need**. NeurIPS, 2017.
35. Steven Davis, Paul Mermelstein. **Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences**. IEEE Transactions on Acoustics, Speech, and Signal Processing, 1980.

## 9.5. Appendix: ký hiệu

| Ký hiệu | Ý nghĩa |
| --- | --- |
| `D(n)` | Dataset gồm n ảnh sản phẩm trong catalog. |
| `q` | Query của người dùng. |
| `K` | Số kết quả trả về. |
| `f(.)` | Feature extractor/encoder. |
| `sim(.)` | Hàm similarity, thường là cosine hoặc inner product sau L2-normalization. |
| ANN | Approximate Nearest Neighbor. |
| JCT | Joint Co-Transformer trong SCALE. |
| SIMCL | Self-harmonized Inter-Modality Contrastive Learning. |
