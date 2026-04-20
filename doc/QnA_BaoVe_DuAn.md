# TÀI LIỆU HỎI ĐÁP & CHUẨN BỊ BẢO VỆ DỰ ÁN (THPT OMR GRADER)

Tài liệu này tập hợp các câu hỏi phản biện khó nhằn thường gặp từ giảng viên/partner khi báo cáo hệ thống Xử Lý Ảnh. Nó sẽ giúp bạn có sẵn các câu trả lời mang tính chuyên môn cao nhưng được giải thích cực kỳ nhạy bén.

---

### Q1: Code xử lý độ phân giải ảnh quét đa dạng như thế nào? Lỡ tôi chụp bằng điện thoại ảnh to ảnh nhỏ thì sao?
**Trả lời:** Dạ thưa thầy/cô, hệ thống không dùng các kích thước đo bằng Pixel cứng nhắc (hardcode) kiểu như "khoảng cách tính bằng 150 pixel". Hầu như toàn bộ thuật toán trong file `config.py` và luồng chạy đều sử dụng **Tỷ Lệ (Ratio)**. 
Ví dụ: Thước đo đường Morphology được tính bằng `0.015 * Chiều rộng ảnh`. Nhờ vậy, bức ảnh độ phân giải thấp (1000px) hay độ phân giải 4K (~7000px) thì tỷ lệ quét khung hình, tính vị trí Part I, II, III đều giữ nguyên tính chính xác hình học không phụ thuộc vào độ sắc nét. 
Bên cạnh đó, việc lọc ra các vùng kết quả (box_grouping) sử dụng một ngưỡng sàn diện tích linh hoạt giúp cứu vãn tất cả các form thiết kế dù bị mờ do scan điện thoại.

### Q2: Điều gì xảy ra nếu học sinh tẩy xóa đáp án, tô 2 ô nhưng có 1 ô đậm 1 ô mờ?
**Trả lời:** Thuật toán chấm điểm của hệ thống (evaluate_grid_fill) sử dụng vòng lặp kiểm tra ranh giới (Threshold Gap). Nó không lấy một mức cố định (ví dụ độ xám < 180) để kết luận là có tô. Thay vào đó, bộ decode Digit tính toán khoảng cách "Mean-Darkness" giữa ô Đen Nhất và ô Đen Thứ Hai. Nếu khoảng cách (sec_gap) này đủ xa (lớn hơn 4.0 px), chứng tỏ học sinh tô rất dứt khoát. Nếu khoảng cách rất thấp sát nhau (như 2.5 px), chứng tỏ đó là vệt tẩy nham nhở hoặc bị mờ mịt; hệ thống sẽ trả về dấu `?` và thông báo lỗi để giáo viên can thiệp bằng tay. Điều này đảm bảo thà "không tự chắc chắn báo tay, còn hơn là chấm sai oan cho thí sinh".

### Q3: Tính năng Affine Transform (Nắn thẳng ảnh) trong dự án có ý nghĩa gì? Bạn xài nó lúc nào?
**Trả lời:** Khi scan phiếu vào máy, tờ giấy thường bị xoay nghiêng hoặc lệch góc (Perspective distortion). Việc này khiến lưới tọa độ lưới chấm điểm (Grid) bị chệch ra ngoài bong bóng vẽ.
Dự án sẽ tiến hành tìm kiếm **4 ô vuông đen đánh dấu vị trí ở 4 góc** của tờ phiếu thiết kế. Dùng 4 tọa độ neo này, kết hợp với hàm `cv2.getPerspectiveTransform` để tạo ma trận xoay, sau đó "kéo giãn" 4 góc tờ phiếu lệch về đúng form chữ nhật chuẩn xác. Hệ thống kích hoạt nắn thẳng Fallback khi điểm đánh giá tìm kiếm Group Box ở base rơi xuống điểm 0 (do bị xoay nên khung thẳng thất bại). Quá trình này giúp hệ thống linh hoạt, chỉ tốn thời gian xoay ở những ảnh thật sự méo.

### Q4: Nếu quy mô thi là hàng ngàn phiếu, tốc độ chương trình có đáp ứng nổi không?
**Trả lời:** Khác với các hệ thống AI OCR tiên tiến cần thời gian infer (suy luận) mất vài giây mỗi ảnh trên các GPU mắc tiền. Hệ thống của em chạy thuần túy bằng các phép toán ma trận NumPy và OpenCV Morphology bằng thuật toán C++ biên dịch ngầm. Do đó, thời gian tính toán trung bình của một frame mất chưa tới ~0.3 - 0.8 giây tùy cấu hình, lại rất nhẹ CPU. Các thao tác đều được tính toán theo lô (Batch Processing arrays). Khi triển khai ở Production, ta hoàn toàn có thể nhồi hệ thống này lên ProcessPoolExecutor để chạy song song (Parallel) 100 ngàn phiếu thi cùng lúc trong vài chục phút cực kì tiết kiệm chi phí cấu hình máy.

### Q5: Tại sao phần xử lý khung, bạn phải loại trừ những khung nhỏ (Deduplication)?
**Trả lời:** Trong lúc Morphology (Tìm dòng kẻ), vì mực in có thể bị loang lổ bởi nét bút vạch đè lên, khiến các đường gióng ngang và dọc tạo thêm rất nhiều lưới nhỏ phụ dính ở cạnh phiều. Việc chẻ nhỏ các khung sẽ tạo ra các đốm gây nhiễu, làm sai lệch hàm lọc ngoại suy (Percentile) hay giá trị trung bình Median. Em loại bỏ và lọc diện tích từ sớm (chỉ giữ các block chứa trên 5000 units area) để bảo vệ mảng tọa độ khỏi việc nhận dạng sai lưới nhỏ thành một Part thi bự.

### Q6: Form phiếu THPT 2025 có cấu trúc phức tạp (Có phần chọn trắc nghiệm Đúng/Sai liên đới nhau). Hệ thống giải quyết thế nào?
**Trả lời:** Em xử lý việc này bằng cách sử dụng Tọa độ lưới Offset linh động. Ở Part II (Phần đúng sai), thay vì tạo một lưới 1 khối khổng lồ, thuật toán cấu hình hàm `extract_grid_from_boxes_variable_offsets`. Nhờ vậy em xây dựng được 4 bảng điểm ghép lại. Mỗi câu là 1 chuỗi 4 bong bóng (a,b,c,d). Ở Part III (Điền số ngắn), thuật toán có tính năng `row_col_patterns` để bỏ qua ô dấu âm/dương ở hàng đầu và chỉ tính điểm cho bong bóng thập phân riêng biệt, khớp cấu trúc vật lý của phiếu 2025.
