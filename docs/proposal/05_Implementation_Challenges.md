# 05. Các thách thức thực hiện

## 5.1. Cách dùng thuật ngữ trong proposal

Mục 02 định nghĩa bốn gap: Sensory, Semantic, Context-Query và Model Gap. Mục này không tạo thêm một bộ gap khác; mỗi mục bên dưới chỉ mô tả cách một gap xuất hiện khi triển khai hệ thống. Ngoài bốn gap, Mục 5.6 là ràng buộc hệ thống về quy mô và cập nhật catalog.

## 5.2. Sensory Gap: ảnh query khác ảnh catalog

Ảnh query thường do người dùng chụp hoặc trích từ mạng xã hội, nên có thể bị nén, mờ, thiếu sáng, chụp nghiêng, che khuất hoặc có nhiều vật thể. Ngược lại, ảnh catalog thường có nền sạch và sản phẩm chiếm phần lớn khung hình. Đây là Sensory Gap, còn được gọi là *domain gap* trong bối cảnh query và catalog thuộc hai phân bố ảnh khác nhau.

Ví dụ, cùng một đôi giày có thể được catalog chụp trên nền trắng, còn query là ảnh chụp ngoài đường bị tối và chỉ thấy một phần thân giày. Nếu embedding học quá phụ thuộc vào nền hoặc ánh sáng catalog, kết quả retrieval sẽ lệch. Chất lượng ảnh tham chiếu vì vậy cần thể hiện rõ sản phẩm, ít vật thể gây nhiễu và có nhiều góc nhìn bổ sung [37].

## 5.3. Semantic Gap: giống ảnh nhưng sai sản phẩm

Trong e-commerce, nhiều candidate có hình dáng tổng thể rất giống nhau nhưng khác ở logo, màu sắc, dung tích, kích thước, model, mã phiên bản hoặc vật liệu. Các chi tiết này có thể nhỏ hơn background hay bố cục, nhưng lại quyết định sản phẩm có phù hợp với người mua hay không.

Ví dụ, hai ốp lưng điện thoại có thể có cùng màu và họa tiết nhưng một chiếc dành cho iPhone 14, chiếc còn lại dành cho iPhone 15. Visual similarity cao không đủ để kết luận hai sản phẩm thay thế được nhau. Thách thức rõ hơn khi query chỉ là ảnh crop một bộ phận, còn candidate là ảnh đầy đủ có background hoặc nhiều item; TIGER-FG gọi sự lệch mức chi tiết này là *granularity disparity* [38].

## 5.4. Context-Query Gap: query thiếu ngữ cảnh và metadata có nhiễu

Query thường chỉ là ảnh, trong khi candidate có thể có thêm title, category, brand và bảng thuộc tính. Sự không cân xứng thông tin này được TIGER-FG gọi là *modality disparity* [38]; trong proposal này, nó được xem là biểu hiện của Context-Query Gap: hệ thống phải quyết định khi nào metadata giúp làm rõ nhu cầu và khi nào nó gây nhiễu.

Ví dụ, ảnh một chiếc túi không cho biết người dùng muốn đúng mẫu, cùng brand hay chỉ cùng kiểu dáng. Nếu query có thêm text như “da thật” hoặc “cho laptop 14 inch”, metadata catalog có thể giúp xếp hạng chính xác hơn. Ngược lại, title quảng cáo quá ngắn, brand/category thiếu hoặc thuộc tính ghi không thống nhất sẽ làm semantic signal kém tin cậy.

## 5.5. Model Gap: kiến thức model không bao quát mọi category

Model chỉ học tốt những loại sản phẩm và thuộc tính xuất hiện đủ trong dữ liệu huấn luyện. Nếu training chủ yếu chứa giày dép và quần áo, embedding có thể phân biệt tốt trong hai nhóm này nhưng không biểu diễn chính xác cặp, ba lô hoặc trang sức. Đây là Model Gap: giới hạn kiến thức theo độ bao phủ category của model, không phải lỗi region extraction.

M5Product giúp giảm gap vì có 6.232 category và dữ liệu đa phương thức từ e-commerce thực tế. Text, table, video và audio cũng bổ sung tín hiệu cho các sản phẩm mà ảnh đơn lẻ khó mô tả. Tuy nhiên, M5Product không bảo đảm model sẽ đúng với mọi loại hàng: category long-tail, sản phẩm mới và category ngoài training vẫn có thể có chất lượng kém. Vì vậy, Model Gap được đánh giá bằng Precision@K/Recall@K và failure analysis tách theo nhóm category, thay vì chỉ dùng một metric trung bình toàn bộ dataset.

## 5.6. Ràng buộc hệ thống: quy mô catalog và dữ liệu động

Đây không phải một gap về dữ liệu hay mô hình, mà là ràng buộc vận hành. Chi phí exact search tăng tuyến tính theo số embedding, trong khi catalog e-commerce có thể gồm từ hàng chục nghìn đến hàng triệu product entry. Hệ thống phải cân bằng độ trễ với chất lượng xếp hạng giữa exact search và approximate nearest-neighbor search.

Catalog cũng thay đổi liên tục: sản phẩm mới được thêm, ảnh được thay thế, sản phẩm hết bán và metadata được hiệu chỉnh. Các cập nhật này có thể làm index, embedding và mapping metadata không đồng bộ. Vì vậy, hệ thống cần quy trình cập nhật tăng dần hoặc tái tạo index theo lô, đồng thời benchmark lại sau các đợt thay đổi lớn [37].
