# 00. Introduction

## Topic

**Visual Product Image Search in E-commerce**

Phương pháp trình bày: trích xuất đặc trưng bằng SCALE và indexing bằng Faiss HNSW.

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
| 0.3 | 2026-08-19 | Trần Hải Đức, Trần Hoàng Nam | Căn chỉnh proposal theo slide: bài toán multimodal retrieval, subset M5Product 10.000 mẫu, SCALE + Faiss HNSW, tái xếp hạng thuộc tính. |

## Scope

Đề tài xây dựng hệ thống **multimodal retrieval** cho sản phẩm thương mại điện tử, không giới hạn ở image retrieval thuần túy. Query và mỗi product entry trong catalog đều có dạng `(Image, Text, Table, Video, Audio)`; các modality có thể thiếu, với điều kiện query tối thiểu có Image hoặc Video.

SCALE tạo embedding thống nhất cho query và cho từng mẫu catalog. Offline, embedding catalog được đánh chỉ mục bằng **Faiss HNSW**. Online, embedding query được so khớp trên chỉ mục để lấy tập ứng viên, sau đó **tái xếp hạng bằng thuộc tính** (siêu danh mục, danh mục/thương hiệu, thông số) trước khi trả top-K.

Proposal mô tả bối cảnh, đặc điểm dữ liệu sản phẩm thương mại, dataset M5Product và cách chọn subset, phát biểu bài toán, thách thức modality interaction/noise, related works, phương pháp SCALE + HNSW, hướng cải tiến tái xếp hạng, tiêu chí đánh giá và tài liệu tham khảo.
