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
