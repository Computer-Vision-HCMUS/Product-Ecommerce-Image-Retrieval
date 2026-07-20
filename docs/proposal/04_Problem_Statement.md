# 04. Problem Statement

## 4.1. Mục tiêu

Mục tiêu là xây dựng hệ thống retrieval cho e-commerce: với catalog gồm các product entry đa phương thức và một ảnh query, hệ thống trả về top-K product entry phù hợp nhất. Text hoặc bảng thuộc tính ngắn, nếu đi kèm query, được dùng làm ngữ cảnh bổ sung chứ không mở rộng phạm vi thành video/audio query.

## 4.2. Formal Definition

| Thành phần | Mô tả |
| --- | --- |
| **Input** | **Catalog D(n)**: n product entry, mỗi entry có ảnh đại diện và metadata/modality sẵn có.<br>**Query**: ảnh; có thể kèm text hoặc bảng thuộc tính ngắn. |
| **Output** | Danh sách top-K product entry phù hợp, kèm ảnh đại diện và metadata. |

Ký hiệu:

- `D = {x_1, x_2, ..., x_n}` là catalog product entry; mỗi `x_i` gồm ảnh đại diện và các modality/metadata có sẵn.
- `q = (q_img, q_ctx)` gồm ảnh query `q_img` và ngữ cảnh tùy chọn `q_ctx` (text/table).
- `f_q(.)` và `f_c(.)` là query encoder và catalog-entry encoder.
- `sim(f_q(q), f_c(x_i))` là độ tương đồng giữa query và product entry.
- `TopK(q, D)` là K product entry có score cao nhất.

Bài toán:

```text
TopK(q, D) = arg top-K_{x_i in D} sim(f(q), f(x_i))
```

## 4.3. Yêu cầu hệ thống

- Trích xuất embedding đủ giàu để biểu diễn visual và semantic similarity.
- Hỗ trợ ảnh query và ngữ cảnh text/table tùy chọn; xử lý missing modality ở catalog.
- Truy hồi nhanh trên catalog lớn.
- Chống nhiễu query như crop, compression, rotation, watermark hoặc background phức tạp.
- Có metric đánh giá rõ ràng cho representation, retrieval và chất lượng hệ thống.

## 4.4. Output mong muốn

Với mỗi query, hệ thống trả về:

- Danh sách `top-K` product entry kèm ảnh đại diện.
- Similarity score hoặc distance score.
- Product metadata cơ bản để phục vụ UI/ranking sau này.
- Thời gian truy hồi và trạng thái index để phục vụ đánh giá hệ thống.
