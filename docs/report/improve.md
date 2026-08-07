# **ĐỀ XUẤT CẢI TIẾN: QUY TRÌNH TÌM KIẾM VÀ TÁI XẾP HẠNG SẢN PHẨM**

Giải pháp đề xuất quy trình truy xuất và tái xếp hạng sản phẩm hai giai đoạn nhằm tối ưu hóa đồng thời độ chính xác ngữ nghĩa của kết quả trả về và tốc độ tìm kiếm trên tập dữ liệu quy mô lớn. Hệ thống dung hòa giữa độ tương đồng thị giác từ mô hình nền tảng SCALE và mức độ trùng khớp thuộc tính phân cấp, giúp tự động linh hoạt điều chỉnh theo kịch bản truy vấn của người dùng.

* Lọc thô tốc độ cao: Sử dụng chỉ mục đồ thị Faiss HNSW để thu hẹp nhanh không gian tìm kiếm hàng triệu sản phẩm với độ trễ thấp.
* Tái xếp hạng tinh chỉnh: Dung hòa điểm tương đồng vector thị giác từ mô hình SCALE với điểm trùng khớp thuộc tính (Siêu danh mục, Danh mục, thông số).
* Linh hoạt kịch bản đầu vào: Tự động thích ứng khi người dùng tìm kiếm chỉ bằng ảnh hoặc kết hợp cả ảnh và văn bản.
* Tùy chỉnh theo ngành hàng: Cho phép linh hoạt điều chỉnh trọng số ưu tiên thị giác hoặc thuộc tính theo đặc thù từng nhóm sản phẩm.
* Khắc phục hạn chế cốt lõi: Loại bỏ lỗi thiên lệch thị giác, xử lý triệt để hiện tượng lệch phân cấp ngành hàng và bù đắp thông tin thuộc tính bị thiếu.

## **II. CHI TIẾT QUY TRÌNH 3 GIAI ĐOẠN**

### **1. Giai đoạn 1: Lọc thô không gian Vector**

Giai đoạn lọc thô đảm nhận nhiệm vụ thu hẹp không gian tìm kiếm từ hàng triệu sản phẩm xuống tập ứng viên kích thước nhỏ ($N$ sản phẩm) với độ trễ thấp.

* Đầu vào: Vector biểu diễn thị giác $\mathbf{v}_{query} \in \mathbb{R}^d$ được trích xuất từ ảnh truy vấn thông qua mô hình trích xuất đặc trưng SCALE.
* Cơ chế xử lý: Thực hiện truy vấn tìm kiếm hàng xóm gần nhất xấp xỉ trên chỉ mục đồ thị Faiss HNSW, sử dụng độ đo tương đồng Cosine (hoặc tích trong trên không gian vector đã qua chuẩn hóa $L_2$).
* Đầu ra: Tập $N$ sản phẩm ứng viên $C = \{p_1, p_2, \dots, p_N\}$ kèm theo điểm tương đồng thị giác tương ứng $S_{emb}(p_i) \in [0, 1]$.

### **2. Giai đoạn 2: Tái xếp hạng tinh chỉnh**

Để giải quyết vấn đề thiên lệch thị giác, bước tái xếp hạng tính toán lại điểm số tổng hợp $S_{tong}$ theo hai kịch bản đầu vào của người dùng.

#### **2.1. Kịch bản A: Truy vấn Đa phương thức (Ảnh và Văn bản)**

Người dùng tải lên ảnh kèm câu truy vấn văn bản chứa các thông tin như: Siêu danh mục ($SD_{truy\_van}$), Danh mục ($D_{truy\_van}$) hoặc Thông số / Từ khóa bổ trợ ($TS_{truy\_van}$).

* Tính điểm thuộc tính ($S_{thuoc\_tinh}$):

$$S_{thuoc\_tinh} = \alpha \cdot \mathbb{I}\left(SD_{truy\_van} = SD_{ung\_vien}\right) + \beta \cdot \mathbb{I}\left(D_{truy\_van} = D_{ung\_vien}\right) + \gamma \cdot \text{Jaccard}\left(TS_{truy\_van}, TS_{ung\_vien}\right)$$

Trong đó:
* $\mathbb{I}(\cdot)$: Hàm chỉ thị, nhận giá trị $1$ nếu điều kiện đúng, và $0$ nếu điều kiện sai hoặc trường dữ liệu tương ứng bị thiếu.
* $\text{Jaccard}(A, B) = \frac{\vert{}A \cap B\vert{}}{\vert{}A \cup B\vert{}}$: Độ đo mức độ trùng khớp giữa tập từ khóa thông số của truy vấn và sản phẩm ứng viên.
* $\alpha, \beta, \gamma$: Các trọng số thành phần thỏa mãn $\alpha + \beta + \gamma = 1$ và $\alpha > \beta$ (ưu tiên sự trùng khớp ở cấp Siêu danh mục trước khi xét tới Danh mục con).

* Tính điểm tổng hợp ($S_{tong}$):

$$S_{tong} = \lambda \cdot S_{emb} + (1 - \lambda) \cdot S_{thuoc\_tinh}, \quad \text{với } \lambda \in [0, 1]$$

#### **2.2. Kịch bản B: Truy vấn Đơn phương thức (Chỉ có Ảnh)**

Người dùng chỉ cung cấp ảnh đầu vào. Hệ thống áp dụng cơ chế Phản hồi độ liên quan giả định để suy luận phân cấp ngành hàng từ $N$ ứng viên ban đầu.

* Trích xuất thuộc tính suy luận (Bầu chọn theo đa số): Thực hiện thống kê tần suất xuất hiện trên tập $N$ ứng viên để xác định bộ nhãn đại diện:
  * Siêu danh mục suy luận ($SD^*$): Nhãn Siêu danh mục có tần suất xuất hiện cao nhất trong $N$ ứng viên.
  * Danh mục suy luận ($D^*$): Nhãn Danh mục có tần suất xuất hiện cao nhất trong $N$ ứng viên.

* Tính điểm thuộc tính suy luận ($S_{suy\_luan}$):

$$S_{suy\_luan} = \alpha \cdot \mathbb{I}\left(SD_{ung\_vien} = SD^*\right) + \beta \cdot \mathbb{I}\left(D_{ung\_vien} = D^*\right)$$

(Với ràng buộc $\alpha + \beta = 1$ và $\alpha > \beta$).

* Tính điểm tổng hợp ($S_{tong}$):

$$S_{tong} = \lambda \cdot S_{emb} + (1 - \lambda) \cdot S_{suy\_luan}, \quad \text{với } \lambda \in [0, 1]$$

### **3. Giai đoạn 3: Trích xuất danh sách K kết quả cao nhất**

* Tiến hành sắp xếp giảm dần toàn bộ $N$ sản phẩm ứng viên dựa trên chỉ số điểm tổng hợp $S_{tong}$.
* Cắt chọn đúng $K$ sản phẩm có điểm số cao nhất ($K < N$, thông thường $K = 20 \sim 50$) để hiển thị trực quan lên giao diện người dùng.
