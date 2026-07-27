# Thuật toán chọn mẫu M5Product

Mục tiêu: chọn **100 category**, mỗi category tối đa **200 sản phẩm**, nhưng không để một ngành hàng (ví dụ chăm sóc da mặt) chiếm phần lớn dataset.

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

## 3. Chọn 200 sản phẩm/category

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

Quy trình: giữ 1.000 ứng viên chất lượng cao nhất/category, rồi chọn lần lượt mẫu có `Score` cao nhất. Sau mỗi lần chọn, Diversity được tính lại để hạn chế mẫu trùng lặp.

## 4. Kết quả

Dataset tối đa có **20.000 sản phẩm**. File kết quả lưu category, siêu danh mục, tier và từng thành phần điểm để có thể kiểm tra lại việc chọn mẫu.
