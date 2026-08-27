# Thuật toán chọn mẫu M5Product

Mục tiêu: chọn **50 siêu danh mục**, mỗi siêu danh mục tối đa **200 sản phẩm**, nhưng không để một ngành hàng (ví dụ chăm sóc da mặt) chiếm phần lớn dataset. Mỗi siêu danh mục được chia theo cohort modality cố định để có thể so sánh công bằng.

## 1. Chuẩn bị

Tạo file taxonomy từ metadata:

```powershell
python app\datasets\download-dataset-tools.py --category-counts-only
python app\datasets\classify_super_categories.py
```

Hai lệnh trên tạo `m5product_category_counts.csv`, `m5product_label_taxonomy.csv` và file tổng hợp trong `app/datasets`.

Input chọn mẫu gồm metadata M5Product và taxonomy CSV:

```csv
label,sieu_danh_muc_tieng_viet
Shiseido/资生堂,Chăm sóc da mặt
手饰,Trang sức
```

Chỉ giữ category có ít nhất 200 mẫu để có thể lấy đủ 200 sản phẩm/category.

## 2. Chọn 100 category cân bằng

1. Đếm số mẫu của mỗi `label`.
2. Gom `label` theo **siêu danh mục** từ taxonomy.
3. Trong từng siêu danh mục, sắp xếp category theo số mẫu giảm dần và chia thành 4 nhóm tần suất: `head`, `medium`, `tail`, `rare`.
4. Lấy luân phiên giữa các siêu danh mục cho đến khi đủ:
   - 30 head
   - 40 medium
   - 20 tail
   - 10 rare
5. Giới hạn mặc định: tối đa 3 category từ một siêu danh mục.

Vì vậy, head/tail/rare được hiểu **tương đối trong ngành hàng của nó**, không phải theo tần suất toàn bộ M5Product.

## 3. Chọn 200 sản phẩm/siêu danh mục và cohort modality

Mỗi sản phẩm có điểm:

```text
Score = 0.4 × Completeness
      + 0.3 × TextQuality
      + 0.2 × MerchantScore
      + 0.1 × Diversity
```

- **Completeness:** có title, label, ảnh, video và thuộc tính `pv`.
- **TextQuality:** title đủ thông tin, không quá ngắn hoặc lặp từ.
- **MerchantScore:** `pv` có thông tin merchant/shop/seller và tín hiệu rating hoặc sales.
- **Diversity:** ưu tiên title, merchant và nguồn ảnh khác các mẫu đã chọn trong cùng category.

Quy trình: giữ 1.000 ứng viên chất lượng cao nhất/siêu danh mục, rồi chọn lần lượt mẫu có `Score` cao nhất. Sau mỗi lần chọn, Diversity được tính lại để hạn chế mẫu trùng lặp.

Chỉ các listing có đủ nguồn để tạo năm modality **image, title, pv, video, audio** mới được chọn. Audio được tách từ audio track của video sau khi tải; `audio_feature_manifest.json` xác nhận feature audio thực sự trích xuất được. Từ cùng pool này, script gán cohort có seed cố định:

| Cohort | Tỷ lệ mặc định | Input hiệu dụng |
|---|---:|---|
| `natural_full` | 70% | image + title + pv + video + audio |
| `masked_missing_1_2` | 20% | image và còn 3 hoặc 4 trong title/PV/video/audio; 70% ẩn 1, 30% ẩn 2 |
| `masked_image_only` | 10% | image-only |

Image luôn còn lại vì pipeline SCALE hiện tại bắt buộc ảnh để tạo region feature. Khi thiếu modality, script **ẩn trường input hoặc zero-fill feature** thay vì thay bằng listing vốn thiếu dữ liệu. Media và trường gốc vẫn được lưu dưới `source_*`; nhờ đó có thể dựng ablation khác từ đúng cùng product IDs mà không tải lại.

`selection_protocol.json`, `modality_selection.json`, `masked_modalities`, `modality_source`, `modality_present` và `source_modality_present` là manifest bắt buộc để đối chiếu cohort, seed và mức thiếu modality sau này.

## 4. Kết quả

Dataset mặc định tối đa có **10.000 sản phẩm**. File kết quả lưu siêu danh mục, cohort modality và từng thành phần điểm để có thể kiểm tra lại việc chọn mẫu.

## 5. Khi mất mạng

Mặc định, lỗi kết nối, DNS hoặc timeout chỉ được đánh dấu tại mẫu đó sau số lần retry đã cấu hình; lượt tải vẫn tiếp tục. Dùng `--stop-on-network-error` nếu muốn dừng toàn bộ lượt tải ngay khi mất mạng. Các mẫu đã hoàn thành luôn nằm trong `manifest.jsonl`.
