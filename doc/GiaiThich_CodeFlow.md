# GIẢI THÍCH CHI TIẾT DỰ ÁN (CODE FLOW & ARCHITECTURE)

Dựa vào cấu trúc code và các tài liệu báo cáo, đây là một hệ thống **chấm điểm OMR không sử dụng Deep Learning/AI nặng nề**, mà sử dụng **Computer Vision truyền thống (Toán học/Hình thái học qua OpenCV + Numpy)**. Điều này mang lại tốc độ xử lý nhanh, tiết kiệm tài nguyên và độ chính xác cao đối với bài toán nhận diện form mẫu cố định.

Luồng tổng thể đi từ **UI -> Core xử lý -> Chấm điểm**. Dưới đây là tác dụng cụ thể của từng phần trong dây chuyền.

---

## 1. BỘ NÃO ĐIỀU PHỐI VÀ CẤU HÌNH

* **`src/config.py`**: Trái tim thông số của toàn bộ dự án. Đây là file chứa tất cả các hằng số, tỉ lệ hình học, mức độ sáng tối. 
  - Ví dụ: `MIN_CONTAINER_AREA` (ngưỡng diện tích lọc rác), `DARKNESS_MIN_SECOND_GAP` (khoảng cách xám chống tẩy xoá), hoặc các `%` kích thước dòng/cột.
  - Mọi logic trong code đều được tính toán theo `Tỉ lệ (Ratio)` được định nghĩa tại đây, đảm bảo hệ thống có thể xử lý tốt mọi độ phân giải (từ ảnh thấp 1000px đến ảnh scan 4K) mà vị trí không bị lệch.

* **`app.py`**: Khối giao diện Web (viết bằng nền tảng Streamlit). Đây là nơi người dùng làm việc trực tiếp: tải ảnh lên, tương tác màn hình đồ hoạ và nhận kết quả. Giao diện này sẽ tiếp nhận ảnh và truyền ảnh đó vào lõi xử lý bên trong.

* **`src/pipeline.py`**: Trái tim điều phối vòng đời của tấm phôi. Tập hợp các bước xử lý riêng rẽ bên dưới thành 1 "dây chuyền sản xuất" (Pipeline) duy nhất chạy tuần tự.

---

## 2. DÂY CHUYỀN XỬ LÝ ẢNH (5 BƯỚC CỐT LÕI)

Trong `pipeline.py`, luồng ảnh sẽ lần lượt đi qua các module giải thuật sau:

### Bước 1: Tiền Xử Lý Ảnh (Preprocessing)
* **File phụ trách:** `src/preprocessing.py`
* **Cơ chế & Chức năng:** 
  Ảnh bài thi được chụp bằng điện thoại hoặc scan thường bị bóng mờ và chênh lệch sáng/tối không đều. Hàm này quy đổi ảnh sang định dạng Trắng Đen (Grayscale). Sau đó, nó sử dụng bộ lọc **CLAHE** (Contrast Limited Adaptive Histogram Equalization) phân nhỏ ảnh thành hàng trăm ô vuông để dệt lại viền tương phản đồng đều. Cơ chế này tự động bù sáng những góc khuất, "đánh bay" các vệt bóng mờ do tay người che mất.

### Bước 2: Tìm nét, Dò Khung & Nắn thẳng ảnh (Morphology)
* **File phụ trách:** `src/morphology.py`
* **Cơ chế & Chức năng:** 
  Sử dụng hệ thống `Adaptive Threshold` gạt sạch các tông xám trên mặt giấy nền trắng, làm nổi bật tuyệt đối các nét mực. Sau đó, thuật toán áp dụng một ma trận chập dạng "thước kẻ ảo" lướt tìm những vạch đen thẳng đứng (Vertical) và vạch nằm ngang (Horizontal), ghép giao điểm của chúng lại tạo thành các **Khung Hình Chữ Nhật (Boxes)** trên bài thi.
* **Tính năng tối quan trọng (Nắn thẳng):** Bước này mang cơ chế Fallback sử dụng **Affine Perspective Transform**. Nó dò 4 cụm khối lượng điểm đen vuông góc ở 4 biên rìa giấy. Dùng toán ma trận xoay, nó sẽ uốn lồng 4 góc của tờ giấy bị cong/xiên, kéo nắn lại phiếu thẳng băng như ban đầu.

### Bước 3: Phân Rã và Nhận Diện Bố Cục (Box Grouping)
* **File phụ trách:** `src/box_grouping.py`
* **Cơ chế & Chức năng:** Đây là bộ mặt trí tuệ của logic khối.
  Tiếp nhận hàng trăm Khung đã tìm được ở bước trước, loại trừ sạch rác và nét gạch nguệch ngoạc nhờ kiểm tra ngưỡng diện tích tối thiểu. Sau đó, file áp dụng thuật toán **Heuristics Hình Học**, tự động gióng cao độ (Y-axis) của các box từ trên xuống dưới một cách thần kỳ:
  - Khung trên cùng được chặt đôi dọc lấy hai bảng **Số Báo Danh** (6 cột) và **Mã Đề** (3 cột).
  - Phần thân bài thi được chia tuần tự theo y-axis thành: **Phần I** (gồm 4 khối dọc lớn), **Phần II** (4 khối cách nhịp xen kẽ dạng cấu trúc Đúng/Sai), và **Phần III** (6 khối điền tay con số). Tránh dùng AI hoàn toàn nhằm tăng tối đa sự tiết kiệm độ trễ thao tác.

### Bước 4: Căng Trải Mắt Lưới (Grid Extraction)
* **File phụ trách:** `src/grid_extraction.py`
* **Cơ chế & Chức năng:** 
  Khi đã khoanh vùng được tọa độ của những chiếc Khung Mẹ (vd 1 khối của Phần I), tập lệnh Offset Toán Học trong file này sẽ rải mắt lưới từ tâm và chia đều ra các vùng định vị "Bong Bóng Cần Điền". Việc thiết kế linh động các Biến Offset biến hàm này có khả năng trích xuất chính xác 40 điểm giao ở những form thi 4 nhánh lựa chọn (Trắc nghiệm thường) lẫn dạng ô gộp (Đúng/Sai).

### Bước 5: Chấm Điểm và Chống Tẩy Xoá
* **File phụ trách:** `src/fill_evaluation.py` và `src/digit_decode.py`
* **Cơ chế & Chức năng:** 
  - `fill_evaluation.py` **(Đếm mực):** Áp một chiếc **Mặt Nạ Hình Tròn (Circle Mask)** chính giữa các tâm lưới tọa độ bong bóng lấy được ở bước 4 (cắt đúng bên trong tâm hình tròn tránh tính nhầm vạch viền ô). Phần mềm sẽ rà soát số điểm lượng mực (black pixels). Nếu số điểm mực đen vượt `54%` (ngưỡng hệ số thiết lập trước) diện tích hình học tâm, thuất toán ra quyết định: học sinh **Có Tô**.
  
  - `digit_decode.py` **(Đọc SBD/Mã Đề chống tẩy xóa):** Trong tình huống học sinh tô đáp án xong tẩy không sạch mờ mờ, code sử dụng giải thuật đánh giá **"Khoảng Cách Khoảng Xám Trung Bình" (Mean Darkness Gap)** tại mỗi cột đứng đếm.
    Tha vì kết luận theo một vạch phân giới độ xám, nó so sánh hai bong bóng đậm nhất của cột. Nếu độ chênh lệch "Sec Gap" giữa ô đậm nhất và ô kế tiếp quá sát nhau (`< 4.0`), chỉ báo học sinh tô nguệch ngoạc đè hai số; lập tức nó trả về chữ "?", tạo tín hiệu nhường lại xử lý cho giáo viên, phòng tránh tối đa việc chấm sai điểm thí sinh. Trái lại, nếu ô cực trị chênh vượt xa thông số này, mới ghi nhận đúng con số quét. 

---
**Tổng kết:** Toàn bộ quá trình là chuỗi phép toán vector đồ sộ của `Numpy` vận hành ngầm, không đòi hỏi Card đồ họa (GPU). Hệ thống sẵn sàng phân luồng Multi-threading chấm ngàn bài trong thời gian cực thấp với độ ổn định vô địch.
