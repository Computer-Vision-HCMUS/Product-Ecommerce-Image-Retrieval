# 04. Problem Statement

## 4.1. Mục tiêu

Mục tiêu là xây dựng hệ thống retrieval cho e-commerce: với catalog gồm các product entry đa phương thức và một ảnh query, hệ thống trả về top-K product entry phù hợp nhất. Text hoặc bảng thuộc tính ngắn, nếu đi kèm query, được dùng làm ngữ cảnh bổ sung chứ không mở rộng phạm vi thành video/audio query.

## 4.2. Formal Definition

| Thành phần | Mô tả |
| --- | --- |
| **Input** | **Catalog D(n)**: n product entry, mỗi entry có ảnh đại diện và metadata/modality sẵn có.<br>**Query**: một ảnh có chứa sản phẩm cần tìm; có thể kèm text hoặc bảng thuộc tính ngắn. |
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

### 4.2.1. Ảnh query là gì?

`q_img` là ảnh do người dùng cung cấp, trong đó sản phẩm cần tìm xuất hiện toàn bộ hoặc một phần. Ảnh query có thể thuộc một trong các tình huống sau:

- **Ảnh chụp thực tế:** người dùng chụp sản phẩm ở nhà, ngoài trời hoặc trong cửa hàng; ảnh có thể thiếu sáng, nghiêng, bị che một phần hoặc có background phức tạp.
- **Ảnh từ media trực tuyến:** ảnh/screenshot lấy từ mạng xã hội, video, quảng cáo hoặc website; ảnh có thể bị nén, watermark, crop hoặc chứa text overlay.
- **Ảnh sản phẩm đã crop:** người dùng cắt riêng chiếc túi, đôi giày hoặc món đồ cần tìm từ một ảnh có nhiều object.
- **Ảnh catalog:** người dùng lưu lại ảnh sản phẩm từ một sàn thương mại điện tử khác; trường hợp này thường sạch hơn nhưng có thể khác góc chụp so với catalog của hệ thống.

Ảnh query không bắt buộc là ảnh studio và không cần trùng chính xác với ảnh catalog. Điều kiện tối thiểu là sản phẩm hoặc chi tiết nhận diện quan trọng của sản phẩm phải nhìn thấy được. Video/audio trong M5Product được dùng để làm giàu product entry ở catalog, không phải input query chính của proposal.

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
