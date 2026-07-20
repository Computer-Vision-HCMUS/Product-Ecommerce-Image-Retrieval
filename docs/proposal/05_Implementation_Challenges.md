# 05. Các thách thức thực hiện

## 5.1. Bản chất của bài toán

Truy vấn ảnh trong thương mại điện tử không chỉ là bài toán tìm hai ảnh có nội dung gần nhau. Một ảnh query có thể biểu đạt nhu cầu tìm đúng một SKU (*Stock Keeping Unit* — mã định danh của một đơn vị hàng hóa/biến thể cụ thể), một biến thể của sản phẩm hoặc một sản phẩm thay thế có cùng thuộc tính. Trong khi đó, catalog chứa ảnh, title, thuộc tính cấu trúc và nhiều hình thức trình bày khác nhau. Vì vậy, hệ thống phải đối mặt đồng thời với sai khác dữ liệu, sự mơ hồ về mục tiêu truy vấn và áp lực phục vụ ở quy mô lớn [36]--[38].

## 5.2. Khoảng cách giữa ảnh query và ảnh catalog

Ảnh query thường do người dùng chụp hoặc trích từ nội dung trên mạng xã hội. Các ảnh này có thể bị nén, mờ, thiếu sáng, chụp nghiêng, che khuất hoặc có nhiều vật thể trong cùng khung hình. Ngược lại, ảnh catalog thường được chuẩn hóa, có nền sạch, góc chụp ổn định và sản phẩm chiếm phần lớn diện tích ảnh. Sự khác biệt này tạo ra *domain gap*: đặc trưng học từ ảnh catalog không nhất thiết còn ổn định trên ảnh query thực tế.

Chất lượng ảnh tham chiếu cũng không đồng đều. Ảnh có sản phẩm quá nhỏ, có người mẫu hoặc vật thể phụ, thiếu các góc nhìn quan trọng hay có nền gây nhiễu đều làm giảm khả năng nhận diện. Tài liệu triển khai product search cho thấy hiệu quả phụ thuộc đáng kể vào việc sản phẩm được thể hiện rõ, ít vật thể gây nhiễu và có nhiều góc nhìn bổ sung [37].

## 5.3. Tính phân biệt thuộc tính chi tiết

Trong e-commerce, nhiều candidate có hình dáng tổng thể rất giống nhau nhưng khác ở các chi tiết có giá trị quyết định đối với người mua: logo, khóa kéo, texture, màu sắc, dung tích, kích thước, mã phiên bản hoặc vật liệu. Các khác biệt này thường nhỏ hơn nhiều so với thay đổi về background, ánh sáng hoặc bố cục.

Thách thức này đặc biệt rõ khi query chỉ là ảnh crop một bộ phận của sản phẩm, còn candidate là ảnh đầy đủ có background hoặc có nhiều item. TIGER-FG mô tả hiện tượng này là *granularity disparity*: biểu diễn toàn cục có thể bị chi phối bởi vùng nổi bật nhưng không liên quan đến đối tượng cần tìm [38]. Do đó, một kết quả có visual similarity cao vẫn có thể sai về biến thể hoặc thuộc tính sản phẩm.

## 5.4. Không đối xứng modality và nhiễu metadata

Query và candidate thường không có cùng loại thông tin. Người dùng có thể chỉ gửi một ảnh, trong khi candidate được mô tả bằng ảnh, title, category, brand và bảng thuộc tính. Ngược lại, query có thể có text nhưng candidate lại thiếu metadata. Sự không đối xứng này được gọi là *modality disparity* trong image-to-multimodal retrieval [38].

Metadata catalog cũng không luôn đáng tin cậy. Title có thể ngắn hoặc tối ưu cho quảng cáo thay vì mô tả; cùng một thuộc tính có nhiều cách ghi; brand/category bị thiếu; và một số record không có video, audio hoặc bảng thông số. Vì vậy, semantic signal từ metadata có thể vừa hỗ trợ phân biệt sản phẩm, vừa tạo thêm nhiễu cho retrieval.

## 5.5. Quy mô catalog và tính động của dữ liệu

Chi phí tìm kiếm exact tăng tuyến tính theo số lượng embedding, trong khi catalog e-commerce có thể gồm từ hàng chục nghìn đến hàng triệu ảnh. Hệ thống phải duy trì độ trễ thấp nhưng vẫn giữ được chất lượng xếp hạng đủ tốt; đây là trade-off cố hữu giữa exact search và approximate nearest-neighbor search.

Catalog cũng thay đổi liên tục: sản phẩm mới được thêm, ảnh được thay thế, sản phẩm hết bán và metadata được hiệu chỉnh. Các cập nhật này có thể làm index, embedding và mapping metadata không đồng bộ. Vì vậy, hệ thống cần có quy trình cập nhật tăng dần hoặc tái tạo index theo lô, đồng thời benchmark lại sau các đợt thay đổi lớn của catalog [37].
