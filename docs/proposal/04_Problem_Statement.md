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
