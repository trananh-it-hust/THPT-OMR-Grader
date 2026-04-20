# BÁO CÁO THUYẾT MINH DỰ ÁN: BỘ CHẤM THI TRẮC NGHIỆM THPT (OMR GRADER)

Tài liệu này được biên soạn để giúp bất kỳ ai (kể cả không quá chuyên sâu về code) cũng có thể nắm bắt mạch lạc cách chương trình hoạt động từ đầu đến cuối. Đây là xương sống để bạn trình bày báo cáo đồ án, khóa luận hoặc giải trình với hội đồng.

---

## 1. GIỚI THIỆU TỔNG QUAN
**Mục tiêu:** Xây dựng một phần mềm tự động đọc và chấm điểm phiếu trả lời trắc nghiệm chuẩn kì thi THPT. 
**Công nghệ lõi:** Python, OpenCV, NumPy. 
**Cách tiếp cận:** Hệ thống **không** sử dụng Trí tuệ Nhân tạo (Machine Learning/AI) nặng nề mà đi theo hướng **Computer Vision truyền thống (Xử lý ảnh bằng Hình thái học và Hình học không gian)**. Điều này giúp code chạy cực nhanh, nhẹ nhàng, không cần card đồ họa (GPU) và dễ dàng truy vết để gỡ lỗi cụ thể từng pixel.

---

## 2. QUY TRÌNH HOẠT ĐỘNG CỦA HỆ THỐNG (PIPELINE)
Toàn bộ luồng mã nguồn chạy qua 5 bước chính như một nhà máy nắp ráp tự động.

### Bước 1: Tiền Xử Lý (Preprocessing)
*Khó khăn:* Ảnh quét từ máy scan hoặc điện thoại thường bị bóng mờ, tối góc, hoặc ánh sáng không đều.
*Giải pháp của code:* 
- Nạp ảnh vào và chuyển sang ảnh xám (Grayscale).
- Sử dụng thuật toán **CLAHE** (Contrast Limited Adaptive Histogram Equalization). Bạn cứ hiểu đơn giản là thuật toán này chia bức ảnh ra thành hàng chục khối nhỏ, xem xét từng cụm sáng tối để "giăng" độ tương phản lại cho đều. Những vệt đen bóng mờ do tay che đi sẽ bị đánh bay, làm hiện lên rõ ràng cấu trúc viền giấy.

### Bước 2: Quét Hình Thái Học Tìm Khung Giấy (Morphology)
*Khó khăn:* Làm sao để máy tính hiểu đâu là các bảng điền trắc nghiệm, đâu là chữ in trên giấy?
*Giải pháp của code:*
- Dùng kỹ thuật lọc ảnh **Adaptive Threshold**: Biến bức ảnh thành 2 màu Trắng/Đen tuyệt đối (Trắng là nét mực, Đen là nền giấy).
- Dùng cửa sổ **Morphology** dẹt (như một cây thước kẻ ảo). Cây thước kẻ này sẽ đi lướt dọc từ trên xuống dưới để lượm nhặt tất cả các "đoạn thẳng đứng", và lướt ngang để nhặt các "đoạn thẳng ngang" in trên giấy.
- Cuối cùng, ghép các đoạn dọc và ngang này lại. Giao điểm tạo thành các **Khung chữ nhật (Boxes)** bao bọc từng phần thi.

### Bước 3: Phân Cụm Bố Cục (Box Grouping - Trái Tim Của Logic)
*Khó khăn:* Máy tính đã tìm ra hàng trăm ô chữ nhật trên phiếu. Nhưng ô nào là Số Báo Danh? Ô nào là Mã Đề? Ô nào là trắc nghiệm Phần I, Phần II?
*Giải pháp của code:*
- **Sàng lọc rác:** Đầu tiên, code dọn dẹp các chấm nhiễu và đốm tròn bằng cách giới hạn lại kích thước với biến config `MIN_CONTAINER_AREA`. Các đốm nhỏ sẽ bị loại bỏ, chỉ giữ lại các "Khung chứa lớn". (Đây là chỉnh sửa đắt giá giúp các phiếu độ phân giải thấp vẫn sống khỏe).
- **Nhận diện qua hình học (Heuristics):** Code sắp xếp tọa độ Y (chiều dọc) của các khung lớn từ trên xuống dưới.
  - Khối Phần I của phiếu 2025 luôn có chính xác **4 khung vuông vức nằm cạnh nhau**.
  - Khối Phần II có **4 khối** chia khoảng cách xen kẽ.
  - Khối Phần III có **6 khối**.
  - Phần phía trên cùng hiển nhiên là vùng chứa 2 bảng: **Số báo danh** (dài hơn, 6 cột) và **Mã đề** (ngắn hơn, 3 cột). Máy tính chỉ việc chẻ đôi nửa trên ra là lấy được chính xác.

### Bước 4: Mắt Lưới & Chấm Điểm Đáp Án (Grid Extraction & Fill Evaluation)
*Hoạt động thế nào?*
- Sau khi đã tách được cái khung lớn của Phần I, máy tính sẽ cắt tỉ lệ (offset) để tạo ra tọa độ tâm của 40 "bong bóng" đáp án bên trong đó giống hệt như một tấm lưới tọa độ.
- Để quyết định học sinh có tô vào ô hay không: Thay vì cắt nguyên cái hình vuông nhỏ bao lấy ô (gây nhiễu do dính lẹm viền kẻ khung), code dùng thao tác **Hough Circle/Mask Circle**. Nó ướm một hình tròn nhỏ hoàn hảo vào giữa tọa độ, sau đó chỉ đếm số pixel mực đen nằm bên trong vòng tròn đó. Nếu mực đen lấp đầy trên `54%` diện tích, kết luận là học sinh **Có tô**. 

### Bước 5: Giải Mã Số Báo Danh (Digit Decoding Gap)
*Khó khăn:* Nhiều thí sinh tẩy xóa Số báo danh rất dơ, để lại một lớp chì mờ. Cảm biến thông thường rất dễ đọc nhầm.
*Trí thông minh của code:*
- Chương trình sử dụng chuẩn đọc **Mean Darkness Gap**. Nó khoanh 10 bong bóng sọc dọc tạo thành 1 cột (đại diện từ số 0 đến 9), và tính ra con số đo độ đen của 10 bong bóng này.
- **Không dùng điểm liệt:** Nó không xét một màu xám cố định nào làm chuẩn. Thay vào đó, nó **so sánh**. Nó tìm bong bóng Đen Nhất (hạng 1) và bong bóng Đen Nhì (hạng 2) trong cột. 
- Nó yêu cầu khoảng cách chênh lệch lực nhấn chì (`second_gap`) phải đạt một chỉ số nhất định trở lên. Nếu có một bong bóng đen trội hơn hoàn toàn mọi bong bóng khác (tức học sinh tô dứt khoát 1 lỗ), nó sẽ chấp nhận con số đó. Nếu khe hở giữa 2 bong bóng dưới 4.0, nó biết rằng đó chỉ là vệt xóa lem và trả về dấy `?` để báo lỗi cần con người can thiệp.

---

## 3. TẠI SAO GIẢI PHÁP NÀY LẠI TỐI ƯU HƠN CÁC PHƯƠNG PHÁP KHÁC? (DÀNH CHO BẢO VỆ ĐỒ ÁN)

**1. Tại sao không dùng YOLO / Deep Learning để dò khung?**
* Trả lời: Form chấm thi quy chuẩn của Bộ GDĐT mang tính cấu trúc hình học cực kỳ nghiêm ngặt và không đổi (tọa độ các ô, khoảng cách viền...). Dùng AI cho bài toán này là việc "lấy dao mổ trâu giết gà", AI sẽ đòi hỏi máy chủ tốn chi phí đắt đỏ (GPU), xử lý chậm và tốn dữ liệu huấn luyện khổng lồ. Cách dùng Mathematics / Morphology hiện hành xử lý nháy mắt trên 1 CPU lõi đơn bé nhất, không cần Card đồ họa, mà độ chính xác gần như 100%.

**2. Điều gì cứu vãn các phiếu méo, xoay, hoặc mờ?**
* Trả lời: Trong dự án được trang bị cơ chế nhận diện 4 điểm neo đen (Corner markers) ở 4 góc giấy. Nhờ thuật toán **Affine Perspective Transform**, khi tờ giấy bị cong méo do đặt vào máy Scan xộc xệch, code sẽ tính toán ma trận ma thuật kéo nắn ảnh thẳng băng trở lại trước khi chấm. Đồng thời màng lọc CLAHE liên tục bù sáng cho các file tối.

**3. Khả năng bảo trì và nâng cấp của dự án**
* Trả lời: Hệ thống chia luồng rất rõ ràng. Toàn bộ các hệ số (to nhỏ, nhạy cảm sáng tối) được tách bóc hoàn toàn sang một file `config.py`. Ngày mai nếu Bộ GD&ĐT quyết định đổi kích cỡ tờ giấy từ A4 sang A5, người dùng không cần biết code, chỉ cần vào file text config sửa lại một vài tỉ lệ số học là hệ thống chạy tiếp ngay.
