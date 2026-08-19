# 02. Đặc điểm sản phẩm thương mại điện tử

## 2.1. Khái niệm

Một mẫu sản phẩm thương mại điện tử không chỉ là ảnh đại diện. Trên trang listing, người mua thường thấy đồng thời ảnh, tên/mô tả, bảng thuộc tính, đôi khi video và audio đi kèm. Các tín hiệu này cùng mô tả một SKU nhưng không trùng nội dung: ảnh nói về hình dáng, table nói về brand/chất liệu/kích thước, video nói về cách dùng.

Trong truy vấn, mẫu sản phẩm là đơn vị retrieval: hệ thống so khớp biểu diễn đa phương thức của query với biểu diễn đa phương thức của từng `p_i` trong kho, chứ không so từng ảnh đơn lẻ như image retrieval thuần túy.

## 2.2. Ba tính chất then chốt

### 2.2.1. Dữ liệu có cấu trúc

Một mẫu sản phẩm thường gồm:

- Ảnh: appearance, màu, hình dáng.
- Text: tên sản phẩm, caption, selling point.
- Table: cặp key-value như thương hiệu, chất liệu, kích thước, xuất xứ.
- Video: nhiều góc nhìn, độ đàn hồi, cách sử dụng.
- Audio: lời giới thiệu hoặc âm thanh tách từ video.

Ví dụ đồng hồ Casio trên slide: ảnh mặt đồng hồ đi cùng caption dòng Accent Color EF-130D-1A2 và bảng thuộc tính (thương hiệu, chống nước 100m, độ dày 13mm, xuất xứ Nhật Bản, loại hiển thị kim). Không modality nào tự đủ để định danh sản phẩm.

### 2.2.2. Dữ liệu không toàn vẹn

Không phải listing nào cũng đủ năm modality. Người bán có thể chỉ đăng ảnh và title, thiếu table, video và audio. M5Product phản ánh đúng thực tế này: nhiều mẫu thiếu một hoặc nhiều modality, khoảng 5% là unimodal. Hệ thống không được loại các mẫu thiếu khỏi huấn luyện hay khỏi catalog; nếu loại, mất dữ liệu và retrieval kém đi vì embedding không học được listing thiếu thông tin.

### 2.2.3. Dữ liệu cực kỳ đa dạng

Catalog không giới hạn một ngành hàng. Giày, đồ gia dụng, đồ phòng khách, thực phẩm, đồng hồ, balo đều có thể xuất hiện. Hình dáng tổng thể của nhiều SKU gần nhau (đế giày, mũi giày, dây giày), trong khi khác nhau ở model, màu, chất liệu. Background cũng không đồng nhất: nền trắng studio, nền màu, hoặc môi trường thật.

## 2.3. Hệ quả với retrieval

| Tính chất | Hệ quả kỹ thuật |
| --- | --- |
| Có cấu trúc, nhiều modality | Cần fusion: mỗi nhánh chỉ chứa một phần thông tin sản phẩm. |
| Không toàn vẹn | Cần cơ chế vẫn encode được mẫu thiếu modality, không discard. |
| Đa dạng ngành hàng | Embedding phải generic, không chỉ học tốt một miền như fashion. |

Quy trình điển hình vì vậy gồm: nhận query đa phương thức → preprocess từng modality → trích xuất embedding → so khớp trên catalog → xếp hạng top-K. Đề tài mở rộng bước so khớp ảnh-ảnh thành so khớp embedding đa phương thức, rồi thêm tái xếp hạng thuộc tính ở Mục 08.

## 2.4. Kết luận chuyển tiếp

Product search trên e-commerce không phải image retrieval thuần túy. Dataset và phương pháp phải gần dữ liệu thật: nhiều modality, thiếu cặp, nhiều ngành hàng. Đây là lý do chọn M5Product và SCALE, đồng thời lấy subset 10.000 mẫu vẫn giữ tỉ lệ mẫu đủ/thiếu modality.
