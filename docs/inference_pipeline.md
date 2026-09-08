# Luồng suy luận local và giao diện

## 1. Tổng quan

```text
Ảnh đầu vào
    │
    ├─► YOLOv8n (ONNX) ──► Hộp bao các xe ──► Crop (nới 8%)
    │                                             │
    │                                             ▼
    │                              EfficientNet-B0 (ONNX) ──► Top-5 dòng xe
    │                                             │
    └─► (không thấy xe → phân loại cả ảnh)        ▼
                                        car_specs.csv ──► Thông số + xe tương tự
```

Toàn bộ chạy bằng **ONNX Runtime trên CPU** — không cần PyTorch hay
`ultralytics` trên máy local.

## 2. Các module

| File | Vai trò |
| :--- | :--- |
| `src/cv/detector.py` | Tầng 1 — phát hiện xe bằng YOLOv8 |
| `src/cv/classifier.py` | Tầng 2 — phân loại 196 dòng xe |
| `src/cv/pipeline.py` | Ghép hai tầng, vẽ kết quả |
| `src/rec/recommender.py` | Tư vấn xe theo nhu cầu / tìm xe tương tự |
| `app.py` | Giao diện Streamlit |

## 3. Tầng phát hiện (`detector.py`)

### Định dạng đầu ra của YOLOv8 ONNX

```text
[1, 84, 8400]
     │    └── 8400 ô dự đoán (ứng với ảnh vào 640×640)
     └────── 84 = 4 tọa độ (cx, cy, w, h) + 80 điểm số lớp COCO
```

Hai điểm khác YOLOv5 mà code phải xử lý đúng:

- **Không có điểm objectness riêng.** YOLOv8 bỏ giá trị này (85 → 84);
  điểm số lớp đóng luôn vai trò độ tin cậy.
- **NMS không được nhúng sẵn.** Phải tự lọc hộp trùng nhau — xem hàm
  `_non_max_suppression`.

### Letterbox

Ảnh được resize về khung vuông 640×640 nhưng **giữ nguyên tỷ lệ**, phần
thừa tô màu xám 114 (đúng giá trị YOLO dùng khi huấn luyện). Hàm trả về
thêm `scale`, `pad_left`, `pad_top` để quy đổi tọa độ hộp bao về hệ của ảnh
gốc.

Nếu bỏ qua bước quy đổi này, hộp bao sẽ lệch trên mọi ảnh không vuông.

### Lọc lớp phương tiện

Chỉ giữ 3 lớp COCO, lấy từ cấu hình chính thức của Ultralytics:

| Chỉ số | Lớp |
| ---: | :--- |
| 2 | car |
| 5 | bus |
| 7 | truck |

### Ngưỡng mặc định

| Tham số | Giá trị | Ý nghĩa |
| :--- | ---: | :--- |
| `conf_threshold` | 0.35 | Bỏ hộp có độ tin cậy thấp |
| `iou_threshold` | 0.45 | Ngưỡng coi hai hộp là trùng nhau |
| `max_detections` | 10 | Số xe tối đa xử lý trong một ảnh |

## 4. Tầng phân loại (`classifier.py`)

Tiền xử lý **phải giống hệt lúc đánh giá trong notebook**, nếu không độ
chính xác sẽ tụt mà không có lỗi nào báo ra:

```text
resize 224×224 → BGR sang RGB → chia 255 → chuẩn hóa theo ImageNet
```

Đầu ra là logits, đưa qua `softmax` để thành xác suất. Hàm `_softmax` trừ
giá trị lớn nhất trước khi mũ hóa để tránh tràn số.

Trả về **top-5** thay vì chỉ top-1: với 196 lớp rất giống nhau (ví dụ
`Audi S4 Sedan 2007` và `Audi S4 Sedan 2012`), top-5 hữu ích hơn nhiều.

## 5. Ghép hai tầng (`pipeline.py`)

### Trường hợp không phát hiện được xe

Nếu YOLOv8 không tìm thấy xe nào, pipeline **phân loại toàn bộ bức ảnh**
thay vì trả về rỗng.

Lý do: ảnh đã được cắt sát vào xe (như ảnh trong Stanford Cars) thì tầng
phát hiện thường không còn gì để làm. Giao diện hiển thị cảnh báo cho người
dùng biết đây là trường hợp này.

### Nới rộng hộp bao khi crop

Dùng `padding=0.08`, **khớp đúng với lúc huấn luyện**. Nếu crop sát hơn
hoặc rộng hơn lúc train, mô hình sẽ nhận ảnh khác phân bố đã học và độ
chính xác giảm.

### Đọc ảnh có ký tự tiếng Việt trong đường dẫn

`cv2.imread` không đọc được đường dẫn chứa ký tự Unicode trên Windows. Hàm
`load_image` dùng `np.fromfile` + `cv2.imdecode` để tránh lỗi này.

## 6. Module tư vấn (`recommender.py`)

### Hai cách dùng

| Hàm | Dùng khi nào |
| :--- | :--- |
| `recommend_by_needs` | Người dùng nhập nhu cầu qua form |
| `recommend_similar` | Tìm xe tương tự xe vừa nhận diện |

`recommend_similar` chính là **cầu nối giữa hai module**: nhận diện ra xe X
thì gợi ý ngay các xe cùng tầm.

### Điều kiện lọc là ràng buộc cứng

Xe không thỏa mãn bị **loại hẳn**, không phải bị trừ điểm. Người dùng nói
"tối đa 800 triệu" thì không nên gợi ý xe 1 tỷ dù nó tốt đến mấy.

### Trọng số đặc trưng

| Đặc trưng | Trọng số | Lý do |
| :--- | ---: | :--- |
| Giá bán | 3.0 | Yếu tố quyết định nhất khi mua xe |
| Kiểu dáng | 2.5 | Sedan và SUV phục vụ nhu cầu khác hẳn nhau |
| Số chỗ | 2.0 | Ràng buộc thực tế của gia đình |
| Phân khúc | 2.0 | Phân biệt xe phổ thông và xe sang |
| Tiêu hao | 1.5 | Chi phí vận hành |
| Năm sản xuất | 1.0 | Ảnh hưởng ít nhất |

Các cột số được chuẩn hóa về `[0, 1]` trước khi nhân trọng số, để giá (hàng
nghìn) không lấn át số chỗ (một chữ số).

### Một vấn đề đã phát hiện và sửa

Ban đầu công thức sinh giá chỉ dựa vào (phân khúc, kiểu dáng, năm), khiến
**196 xe chỉ có 87 tổ hợp đặc trưng khác nhau** — 18 xe cùng giá 750 triệu.
Kết quả tư vấn vì thế trông như xếp hạng tùy tiện: nhiều xe cùng điểm
1.000.

Đã sửa bằng hàm `model_variation` trong `scripts/generate_car_specs.py`:
sinh độ lệch riêng cho từng dòng xe dựa trên **hash md5 của tên xe**, nên
kết quả luôn tái lập được chứ không phải số ngẫu nhiên.

Sau khi sửa: **196/196 tổ hợp riêng biệt**.

## 7. Giao diện (`app.py`)

### Chạy

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Mặc định mở tại <http://localhost:8501>.

### Hai tab

**Tab "Nhận diện xe"** — tải ảnh lên hoặc dán URL, hiển thị ảnh có vẽ hộp
bao, top-5 dự đoán kèm thanh xác suất, thông số kỹ thuật và các xe tương
tự.

**Tab "Tư vấn xe"** — form nhập ngân sách, số chỗ, kiểu dáng, mức tiêu hao;
trả về bảng xe phù hợp.

> Tab tư vấn **chạy được ngay cả khi chưa có mô hình**, vì nó chỉ cần
> `data/car_specs.csv`.

### Nạp mô hình một lần

Dùng `@st.cache_resource` để mô hình chỉ nạp một lần rồi dùng lại. Nếu
không, mỗi lần người dùng bấm nút Streamlit sẽ chạy lại toàn bộ script và
nạp lại mô hình từ đĩa.

### Tải ảnh từ URL

Có giới hạn an toàn: kiểm tra `Content-Type` phải là `image/*`, timeout 15
giây, dung lượng tối đa 20 MB.

## 8. Mô hình giả để kiểm thử

Trong lúc mô hình thật còn đang huấn luyện trên Colab, có thể sinh mô hình
giả để chạy thử toàn bộ luồng:

```powershell
.\.venv\Scripts\python.exe scripts/make_dummy_models.py
```

Mô hình giả có **đúng cấu trúc đầu vào/đầu ra** như mô hình thật, nhưng
trọng số ngẫu nhiên nên **kết quả dự đoán vô nghĩa**. Chúng chỉ để kiểm tra
code chạy thông, không dùng để đánh giá độ chính xác.

Khi có mô hình thật, ghi đè lên các file trong `models/` là xong.

## 9. Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

| File | Nội dung |
| :--- | :--- |
| `test_generate_car_specs.py` | Sinh bảng thông số |
| `test_recommender.py` | Logic tư vấn, hàm biến thiên |
| `test_cv.py` | Letterbox, NMS, softmax, crop, pipeline |

Các test cần mô hình sẽ **tự bỏ qua** nếu `models/` còn trống, thay vì báo
lỗi.

## 10. Lấy mô hình YOLOv8 ONNX

Máy local không cài `ultralytics` (kéo theo `torch` ~2,5 GB), nên việc
export YOLOv8 sang ONNX được làm luôn **trong notebook Colab** — nơi đã có
sẵn torch. Xem mục 15 của notebook.

Notebook cũng kiểm tra dạng đầu ra đúng `[1, 84, 8400]` ngay sau khi export,
để không phát hiện sai lệch lúc chạy app.
