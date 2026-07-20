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

Đề tài tập trung xây dựng hệ thống tìm kiếm sản phẩm trong thương mại điện tử với **ảnh là query chính**. Text hoặc bảng thuộc tính ngắn có thể được dùng làm ngữ cảnh bổ sung khi có sẵn. Mỗi product entry trong catalog có thể chứa image, text, table, video và audio; SCALE tạo embedding cho entry, `IndexFlatIP` là exact baseline và Faiss HNSW là index chính. Hệ thống trả về top-K product entry kèm ảnh đại diện và metadata.

Proposal này mô tả động lực, đặc trưng dữ liệu product e-commerce image, dataset M5Product, problem statement, related works, methodology dự kiến, tiêu chí đánh giá, kế hoạch thực hiện và danh mục tài liệu tham khảo.
