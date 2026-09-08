# Công cụ dòng lệnh và đo hiệu năng

## 1. Nhận diện từ dòng lệnh

`scripts/recognize.py` cho phép nhận diện xe **không cần mở giao diện web**.
Hữu ích khi thử nhanh, xử lý hàng loạt, hoặc lấy kết quả để đưa vào báo cáo.

### Cách dùng

```powershell
# Một ảnh
.\.venv\Scripts\python.exe scripts/recognize.py anh_xe.jpg

# Kèm gợi ý xe tương tự
.\.venv\Scripts\python.exe scripts/recognize.py anh_xe.jpg --recommend

# Lưu ảnh đã vẽ hộp bao
.\.venv\Scripts\python.exe scripts/recognize.py anh_xe.jpg --save ket_qua.jpg

# Xử lý cả thư mục
.\.venv\Scripts\python.exe scripts/recognize.py thu_muc_anh/

# Xuất JSON (để xử lý tiếp bằng script khác)
.\.venv\Scripts\python.exe scripts/recognize.py anh_xe.jpg --json
```

### Tham số

| Tham số | Ý nghĩa |
| :--- | :--- |
| `source` | Đường dẫn một ảnh hoặc một thư mục |
| `--save` | Lưu ảnh đã vẽ hộp bao (chỉ khi xử lý một ảnh) |
| `--recommend` | Hiện thêm các xe tương tự |
| `--json` | Xuất JSON thay vì bảng chữ |
| `--top-k` | Số dự đoán mỗi xe (mặc định 5) |

### Ví dụ kết quả

```text
==================================================================
Anh: xe_1.jpg
==================================================================
Xe 1: box=(220,165)-(420,315) conf=90% loai=car
    51.05%  Ferrari 458 Italia Coupe 2012
    13.76%  smart fortwo Convertible 2012
    10.69%  Spyker C8 Coupe 2009
    -> Coupe, 2 cho, 15,07 ty, 14.1 L/100km
    Xe tuong tu:
      - McLaren MP4-12C Coupe 2012 (14,14 ty)
      - Lamborghini Gallardo LP 570-4 Superleggera 2012 (15,95 ty)
```

### Ghi chú kỹ thuật

**Chế độ `--json` không in log.** Để đầu ra luôn là JSON hợp lệ, có thể
đưa thẳng vào `jq` hoặc script khác mà không bị nhiễu bởi dòng log.

**Ghi ảnh qua `imencode`.** Dùng `cv2.imencode` + `write_bytes` thay cho
`cv2.imwrite`, vì `imwrite` không ghi được đường dẫn chứa ký tự tiếng Việt
trên Windows.

## 2. Đo hiệu năng suy luận

`scripts/benchmark_inference.py` đo thời gian từng tầng và cả luồng, sinh
số liệu cho chương đánh giá.

### Cách dùng

```powershell
# Dùng ảnh tổng hợp
.\.venv\Scripts\python.exe scripts/benchmark_inference.py

# Dùng ảnh thật, chạy 100 lần
.\.venv\Scripts\python.exe scripts/benchmark_inference.py --runs 100 --image anh.jpg
```

### Ví dụ kết quả

Đo trên Intel Core i7-10510U (mô hình giả — số liệu chỉ minh họa cấu trúc
báo cáo):

```text
======================================================================
KET QUA (30 lan chay, anh 640x480)
======================================================================
Buoc xu ly                 Trung vi       TB  Do lech      p95      Max
                               (ms)     (ms)     (ms)     (ms)     (ms)
----------------------------------------------------------------------
Tang 1 detect (640px)         11.12    11.22     0.79    12.66    12.71
Tang 2 classify (224px)        7.14     7.06     0.54     7.87     7.93
----------------------------------------------------------------------
Ca luong (end-to-end)         20.14    20.34     1.48    22.33    25.11
======================================================================

PHAN TICH
----------------------------------------------------------------------
  Thong luong        : 49.6 anh/giay
  Thoi gian mot anh  : 20.1 ms
  Ty trong tang 1    : 55%
  Ty trong tang 2    : 35%
  Chi phi con lai    : 1.9 ms (9%) — crop, doi mau, ghep ket qua
======================================================================
```

### Cách đọc số liệu

| Chỉ số | Ý nghĩa |
| :--- | :--- |
| **Trung vị** | Số liệu chính — ít bị nhiễu bởi tiến trình khác |
| Trung bình | Dễ bị kéo lệch bởi vài lần chạy chậm bất thường |
| Độ lệch chuẩn | Mức dao động; càng nhỏ càng ổn định |
| **p95** | 95% số lần chạy nhanh hơn mức này |
| Max | Trường hợp chậm nhất ghi nhận được |

Dùng **trung vị** làm số liệu chính khi báo cáo. Phép đo thời gian trên máy
cá nhân luôn bị nhiễu bởi các tiến trình nền, trung vị phản ánh hiệu năng
điển hình tốt hơn trung bình.

### Cảnh báo mô hình giả

Script tự nhận biết mô hình giả và in cảnh báo. Cách nhận biết là **đếm số
node trong đồ thị ONNX** (mô hình giả chỉ có vài node, mô hình thật có hàng
trăm), chứ **không dựa vào kích thước file** — ma trận trọng số ngẫu nhiên
của mô hình giả cũng lên tới hơn 100 MB.

### Lưu ý khi đưa vào báo cáo

Số liệu phụ thuộc mạnh vào CPU của máy đo. Khi trình bày, cần ghi kèm:

- Model CPU
- Phiên bản ONNX Runtime
- Execution Provider (`CPUExecutionProvider`)
- Kích thước ảnh đầu vào
- Số lần chạy

Script tự in đủ các thông tin này ở đầu kết quả.

## 3. Bảng tổng hợp các script

| Script | Chức năng |
| :--- | :--- |
| `generate_car_specs.py` | Sinh bảng thông số 196 xe |
| `make_dummy_models.py` | Tạo mô hình giả để kiểm thử |
| `build_train_notebook.py` | Sinh notebook huấn luyện |
| `recognize.py` | Nhận diện từ dòng lệnh |
| `benchmark_inference.py` | Đo hiệu năng suy luận |
| `evaluate_model.py` | Sinh báo cáo từ checkpoint |
| `export_recommender_onnx.py` | Export + so sánh module tư vấn |
