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

## 4. CHI TIẾT Ý NGHĨA CÁC HÀM CỐT LÕI (CORE FUNCTIONS)

Dưới đây là các "mắt xích" quan trọng trong mã nguồn mà em cần nắm vững khi bảo vệ đồ án:

### A. Nhóm Xử lý Ảnh & Hình thái học (Morphology)
*   **`detect_boxes_from_morph_lines`**: Hàm thực hiện quét ảnh để tìm các đường kẻ. Nó sử dụng `cv2.getStructuringElement` để tạo các "thanh thước kẻ" dọc và ngang, sau đó dùng toán tử `bitwise_and` để tìm giao điểm. Đây là bước quan trọng nhất để xác định lưới tọa độ.
*   **`detect_black_corner_markers`**: Tìm 4 điểm đen ở góc giấy. Ý nghĩa là để xác định khung tham chiếu (Reference frame), giúp hệ thống biết tờ giấy đang bị xoay hay méo bao nhiêu độ.
*   **`preprocess_clahe`**: Cân bằng ánh sáng cục bộ. Nó giúp xử lý các ảnh bị bóng đổ (ví dụ học sinh chụp ảnh bằng điện thoại thường bị bóng của chính mình che mất một góc phiếu).

### B. Nhóm Tư duy Bố cục (Grouping & Segmentation)
*   **`group_boxes_into_parts`**: Dựa vào danh sách hàng trăm box tìm được, hàm này dùng logic khoảng cách (Spatial grouping) để gom chúng thành 3 cụm: Phần I, II, và III. Nó là bộ não giúp máy tính không bị "loạn" giữa hàng tá ô vuông.
*   **`separate_upper_id_boxes`**: Sử dụng tọa độ X của Phần I làm chuẩn để chia đôi vùng phía trên thành 2 cột: Cột bên trái mặc định là Số báo danh (SBD), cột bên phải là Mã đề.

### C. Nhóm Giải mã & Chấm điểm (Decoding & Grading)
*   **`evaluate_digit_rows_mean_darkness`**: Đây là kỹ thuật "Mean-Darkness". Thay vì nhận diện hình ảnh, nó đo mật độ pixel (density) trong các ô SBD/Mã đề. Nó so sánh độ đen của ô được tô với các ô còn lại trong cùng một cột để đưa ra kết quả. Nếu độ chênh lệch (gap) quá thấp, nó sẽ báo lỗi `?`.
*   **`evaluate_grid_fill_from_binary`**: Chấm điểm các câu trắc nghiệm. Nó sử dụng một "Circular Mask" (màng lọc hình tròn) để chỉ đếm vết chì bên trong vòng tròn đáp án, loại bỏ hoàn toàn nhiễu từ các đường kẻ khung quanh ô.
*   **`extrapolate_missing_rows`**: Hàm ngoại suy. Nếu một hàng bị mờ đến mức máy không tìm thấy box, hàm này sẽ lấy tọa độ của các hàng trên/dưới để "dựng lại" hàng bị mất, đảm bảo không bỏ sót câu trả lời nào.

---

## 5. CẤU TRÚC GIAO DIỆN WEB (STREAMLIT UI)

Giao diện được thiết kế theo hướng thực dụng, giúp giáo viên điều khiển thuật toán một cách trực quan:

1.  **Sidebar (Thanh cấu hình bên trái):**
    *   **Ngưỡng Fill Ratio (Threshold sliders):** Đây là "nút vặn" độ nhạy. Em có thể kéo để thay đổi mức độ ghi nhận một ô là "đã tô". Tính năng này cực kỳ hữu ích vì mỗi loại bút chì (2B, HB) hoặc lực tô của học sinh là khác nhau.
    *   **Chế độ Debug:** Khi bật, hệ thống sẽ trưng bày các ảnh nội bộ (Binary mask, Detected boxes). Đây là công cụ phục vụ việc kiểm thử (Testing) và minh oan khi có phản ánh chấm sai.
2.  **Bảng Kết Quả Tổng Hợp (Batch Gallery):** Hiển thị danh sách tất cả các ảnh đã xử lý. Giáo viên có thể nhìn nhanh các cột "SBD", "Mã đề", "Số câu đã làm" để phát hiện các phiếu lỗi (Status: ERROR) mà không cần mở từng ảnh.
3.  **Khu vực Chi tiết (Detailed View):**
    *   **Tabs (SBD, Mã đề, Phần I/II/III):** Phân loại dữ liệu rõ ràng. Mỗi tab sử dụng các thẻ **Metric** (con số lớn) để hiển thị đáp án đã trích xuất.
    *   **JSON Tab:** Hiển thị dữ liệu thô dưới dạng mã máy. Đây chính là thứ mà chúng ta sẽ lưu vào server hoặc đẩy vào file Excel điểm số.
4.  **Hệ thống Overlay (Vẽ đè):** Máy sẽ vẽ các vòng tròn xanh lá cây trực tiếp lên ảnh gốc để giáo viên đối soát. Nếu máy chấm câu A mà ảnh học sinh tô câu B, giáo viên sẽ thấy ngay sự lệch lạc này.

---

## 6. Ý NGHĨA KẾT QUẢ JSON (OUTPUT DATA)

File JSON là sản phẩm cuối cùng của hệ thống. Hiểu cấu trúc này là chìa khóa để kết nối với các hệ thống quản lý điểm:

```json
{
  "res": {
    "fc": { "1": [0], "2": [1, 2] }, // "fc" = Full Choice (Phần I). Câu 1 chọn A (0), Câu 2 chọn B và C (lỗi tô 2 ô).
    "fc_invalid": ["2"],            // Danh sách các câu bị lỗi ở Phần I.
    "tf": { "1": [1] },              // "tf" = True/False (Phần II). 0: Sai, 1: Đúng.
    "dg": { "1": "123" },            // "dg" = Digits (Phần III). Đáp án tự luận là 123.
    "sbd": "012345",                 // 6 chữ số Số báo danh.
    "mdt": "101",                    // 3 chữ số Mã đề.
    "sbd_invalid": false,            // Nếu là true, nghĩa là hệ thống không tự tin vào kết quả SBD (cần kiểm tra tay).
    "mdt_invalid": false
  }
}
```
*   **Tính toàn vẹn (Integrity):** Mọi câu hỏi đều có mặt trong JSON dù thí sinh có làm hay không (nếu không làm sẽ trả về danh sách trống `[]`).
*   **Hỗ trợ chấm phúc khảo:** Toàn bộ lịch sử "quyết định" của máy (ví dụ độ đen bao nhiêu, tọa độ nào) đều có thể được truy xuất lại từ các metadata đi kèm (không hiện hết trong JSON rút gọn nhưng có trong bộ nhớ cache).

---

## 7. KẾT LUẬN
Dự án này không đơn thuần là "nhận diện hình ảnh", mà là một hệ thống **Xử lý luồng dữ liệu (Data Pipeline)**: biến các nguyên tử pixel thành các bit dữ liệu có ý nghĩa. Với một sinh viên năm 3, việc hiểu rõ các hàm `morphology` và `grouping` trong dự án này sẽ giúp em có nền tảng cực tốt về tư duy thuật toán thực tế.
