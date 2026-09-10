# Tài liệu Kỹ thuật Dự án AI: Hệ thống Nhận diện và Tư vấn Ô tô

## 1. Tổng quan Dự án

Dự án xây dựng một hệ thống Trí tuệ Nhân tạo đa nhiệm, tích hợp khả năng
nhận diện dòng xe ô tô qua hình ảnh và tư vấn dòng xe phù hợp dựa trên nhu
cầu cá nhân hóa của người dùng.

Hệ thống được thiết kế để tối ưu hóa hiệu suất khi vận hành tại môi trường
cục bộ (Local Environment) thông qua định dạng ONNX — toàn bộ khâu suy luận
chạy trên CPU, không cần GPU và không cần cài PyTorch.

> **Lưu ý:** tài liệu này phản ánh **kiến trúc đã triển khai thực tế**. Một
> số quyết định khác với bản đề cương ban đầu; các thay đổi đó được ghi rõ
> kèm lý do tại mục 6.

## 2. Ngôn ngữ Lập trình & Công nghệ Cốt lõi

| Hạng mục | Công nghệ / Ngôn ngữ | Mục đích sử dụng |
| :--- | :--- | :--- |
| Ngôn ngữ chính | Python **3.11** | Toàn bộ logic AI, luồng dữ liệu và xử lý ảnh |
| Thị giác máy tính | YOLOv8, EfficientNet-B0, OpenCV | Phát hiện xe, phân loại dòng xe, tiền xử lý ảnh |
| Huấn luyện | PyTorch, Torchvision | Fine-tune mô hình phân loại (chỉ chạy trên Colab) |
| Định dạng & Tối ưu | ONNX, ONNX Runtime | Chạy độc lập với framework gốc, tăng tốc suy luận |
| Học máy | Scikit-learn, Pandas, NumPy | Module tư vấn, xử lý bảng thông số |
| Giao diện | Streamlit | Web-app cho phép tải ảnh và nhập nhu cầu |

> **Bắt buộc Python 3.11.** `onnxruntime`, `torch` và `opencv-python` chưa
> có wheel cho Python 3.13+. Chi tiết: `environment_setup.md`.

### Phân tách môi trường

| Môi trường | File requirements | Nội dung |
| :--- | :--- | :--- |
| Máy local | `requirements.txt` | Chỉ thư viện để **chạy** (ONNX Runtime) |
| Google Colab | `requirements-train.txt` | Thêm `torch`, `ultralytics`, `skl2onnx` |

Máy local **không cần** cài PyTorch (~2,5 GB) — đây chính là giá trị thực
tế của việc dùng ONNX trong dự án này.

## 3. Kiến trúc Công nghệ Chi tiết

### 3.1. Module Nhận diện xe — kiến trúc 2 tầng

```text
Ảnh đầu vào
    │
    ├─► [Tầng 1] YOLOv8n ONNX ──► Hộp bao xe ──► Crop (nới 8%)
    │                                                │
    │                                                ▼
    │                          [Tầng 2] EfficientNet-B0 ONNX ──► Top-5
    │                                                │
    └─► (không thấy xe → phân loại cả ảnh)           ▼
                                    car_specs.csv ──► Thông số + xe tương tự
```

| Tầng | Mô hình | Huấn luyện |
| :--- | :--- | :--- |
| 1. Phát hiện xe | YOLOv8n pretrained COCO | **Không cần train** |
| 2. Phân loại dòng xe | EfficientNet-B0, 196 lớp | Fine-tune trên Colab |

#### Vì sao tách 2 tầng?

Bộ dữ liệu Stanford Cars **không có bounding box gán nhãn theo từng dòng
xe** — mỗi ảnh chỉ có một hộp bao duy nhất với nhãn là tên dòng xe. Với dữ
liệu này, không thể huấn luyện YOLOv8 nhận diện trực tiếp 196 lớp.

Giải pháp: dùng YOLOv8 pretrained COCO (đã biết các lớp `car`, `bus`,
`truck`) để khoanh vùng, rồi huấn luyện riêng một mô hình phân loại trên
ảnh đã crop. Cách này vừa đúng với dữ liệu sẵn có, vừa tiết kiệm toàn bộ
công sức gán nhãn lại.

#### Tầng 1 — Phát hiện xe

- **Mô hình:** YOLOv8n (bản nhỏ nhất) pretrained trên COCO
- **Lọc lớp:** chỉ giữ `car` (2), `bus` (5), `truck` (7)
- **Đầu vào:** 640×640, letterbox giữ nguyên tỷ lệ ảnh
- **Đầu ra:** `[1, 84, 8400]` — 4 tọa độ + 80 điểm số lớp

Hai đặc điểm của YOLOv8 mà code phải xử lý đúng:

1. **Không có điểm objectness riêng** (85 → 84 so với YOLOv5); điểm số lớp
   đóng luôn vai trò độ tin cậy.
2. **NMS không được nhúng sẵn** trong file ONNX — phải tự cài đặt thuật
   toán loại hộp trùng.

#### Tầng 2 — Phân loại dòng xe

- **Mô hình:** EfficientNet-B0 (~5,3 triệu tham số), pretrained ImageNet
- **Dữ liệu:** Stanford Cars — 16.185 ảnh, 196 lớp
- **Đầu vào:** 224×224, chuẩn hóa theo thống kê ImageNet
- **Đầu ra:** 196 logits → softmax → **top-5** dòng xe

Chọn B0 vì cân bằng giữa độ chính xác và tốc độ suy luận trên CPU sau khi
export ONNX.

Trả về top-5 thay vì chỉ top-1: với 196 lớp rất giống nhau (ví dụ
`Audi S4 Sedan 2007` và `Audi S4 Sedan 2012`), top-5 phản ánh độ hữu ích
thực tế tốt hơn nhiều.

#### Chuyển đổi sang ONNX

```python
torch.onnx.export(
    best_model,
    dummy_input,
    'car_classifier.onnx',
    input_names=['input'],
    output_names=['logits'],
    # Chỉ để batch động; chiều rộng/cao cố định 224.
    dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
    do_constant_folding=True,
)
```

Sau khi export, notebook **đối chiếu kết quả PyTorch và ONNX Runtime** trên
cùng đầu vào, yêu cầu sai lệch tuyệt đối `< 1e-3`.

#### Suy luận

Khởi tạo `InferenceSession` từ `onnxruntime` với `CPUExecutionProvider`,
nạp mô hình vào bộ nhớ và dự đoán trực tiếp từ ma trận ảnh của OpenCV.

### 3.2. Module Tư vấn (Recommendation Engine)

- **Thuật toán:** Content-based Filtering — biểu diễn mỗi xe thành vector
  đặc trưng đã chuẩn hóa, đo khoảng cách Euclid.
- **Dữ liệu:** `data/car_specs.csv` — 196 dòng, các trường giá bán, phân
  khúc, số chỗ ngồi, mức tiêu hao nhiên liệu, tổ chức trong DataFrame.

#### Hai cách sử dụng

| Hàm | Dùng khi nào |
| :--- | :--- |
| `recommend_by_needs` | Người dùng nhập nhu cầu qua form |
| `recommend_similar` | Tìm xe tương tự xe vừa nhận diện |

`recommend_similar` chính là **cầu nối giữa hai module** — nhận diện ra xe
nào thì gợi ý ngay các xe cùng tầm, biến hai module rời rạc thành một hệ
thống thống nhất.

#### Trọng số đặc trưng

| Đặc trưng | Trọng số | Lý do |
| :--- | ---: | :--- |
| Giá bán | 3.0 | Yếu tố quyết định nhất khi mua xe |
| Kiểu dáng | 2.5 | Sedan và SUV phục vụ nhu cầu khác hẳn |
| Số chỗ | 2.0 | Ràng buộc thực tế của gia đình |
| Phân khúc | 2.0 | Phân biệt xe phổ thông và xe sang |
| Tiêu hao | 1.5 | Chi phí vận hành |
| Năm sản xuất | 1.0 | Ảnh hưởng ít nhất |

#### Đóng gói ONNX — đã thực hiện và đo đạc

Đề cương yêu cầu chuyển pipeline Scikit-learn sang ONNX bằng `skl2onnx`.
Dự án đã triển khai **cả hai bản** và đo thời gian suy luận:

| Bản triển khai | Trung vị | Kết luận |
| :--- | ---: | :--- |
| Pandas + NumPy | **0.698 ms** | Nhanh hơn 1.18× |
| ONNX Runtime (KNN) | 0.823 ms | Chậm hơn |

Hai bản cho **kết quả giống hệt nhau**. Với 196 dòng × 16 đặc trưng, chi
phí cố định của ONNX Runtime lấn át phần tính toán thật.

**Kết luận:** ứng dụng dùng bản Pandas. ONNX có lợi ở mô hình nặng (như
tầng phân loại ảnh) chứ không phải mọi bài toán. Phân tích đầy đủ:
`onnx_vs_pandas.md`.

## 4. Dữ liệu

### 4.1. Ảnh — Stanford Cars

16.185 ảnh, 196 lớp (hãng + dòng + kiểu dáng + năm), chia 8.144 train /
8.041 test.

Trang gốc của Stanford **đã bị gỡ bỏ**, khiến cả
`torchvision.datasets.StanfordCars` không tải được nữa. Dự án dùng bản
mirror trên Kaggle: `rickyyyyyyy/torchvision-stanford-cars` — bản này giữ
nguyên các file `.mat` gốc, nghĩa là có **bounding box** (cần để crop) và
**tên lớp chuẩn**.

### 4.2. Bảng thông số xe

Stanford Cars **không cung cấp** giá bán hay mức tiêu hao nhiên liệu. Bảng
`car_specs.csv` được sinh tự động ở hai mức độ tin cậy:

| Mức | Trường | Cách có được | Độ tin cậy |
| :--- | :--- | :--- | :--- |
| Suy luận | hãng, dòng, kiểu dáng, năm, số chỗ | Tách từ tên lớp | **Chính xác** |
| Mô phỏng | giá bán, mức tiêu hao | Công thức xác định | **Mô phỏng** |

> Giá bán và mức tiêu hao là **số liệu mô phỏng**, không phải giá thị
> trường thật. Quy tắc sinh: `car_specs_generation.md`.

## 5. Yêu cầu Cài đặt Môi trường

1. Thiết lập môi trường ảo Python 3.11 để cô lập thư viện:
   `py -3.11 -m venv .venv`
2. Cài các gói phụ thuộc: `pip install -r requirements.txt`
3. Cấu hình Execution Provider trong ONNX Runtime thành
   `CPUExecutionProvider`

Chi tiết và các lỗi thường gặp: `environment_setup.md`.

## 6. Các quyết định khác với đề cương ban đầu

Bảng dưới ghi lại những điểm đã thay đổi trong quá trình triển khai, kèm
lý do. Đây là phần nên trình bày trong báo cáo.

| # | Đề cương ban đầu | Đã triển khai | Lý do |
| ---: | :--- | :--- | :--- |
| 1 | YOLOv8 nhận diện thẳng ra 196 dòng xe | Kiến trúc 2 tầng (detect + classify) | Stanford Cars không có bbox theo từng dòng xe |
| 2 | `dynamic=True` cho mọi trục khi export | Chỉ batch động, cố định 224×224 | Ảnh luôn resize về 224 trước suy luận; trục động chỉ làm ONNX Runtime khó tối ưu |
| 3 | Python 3.9+ | Bắt buộc Python 3.11 | `onnxruntime`/`torch` chưa hỗ trợ 3.13+ |
| 4 | Đóng gói recommender sang ONNX để "đồng bộ kiến trúc" | Dùng bản Pandas, giữ bản ONNX để so sánh | Đo được: Pandas nhanh hơn 1.18×; ONNX không loại bỏ được Pandas khỏi khâu lọc |
| 5 | Streamlit **hoặc** Gradio | Streamlit | Form nhiều trường của module tư vấn dễ dựng hơn |

## 7. Kết quả và Đánh giá

### Chỉ số đánh giá

Báo cáo cả **top-1** và **top-5** accuracy trên tập test 8.041 ảnh.

Kỳ vọng với EfficientNet-B0 + 20 epoch trên Colab free: top-1 khoảng
**80-88%**, top-5 thường trên 95%.

### Công cụ sinh báo cáo

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_model.py
```

Script đọc checkpoint và xuất bảng số liệu từng epoch (định dạng Markdown),
biểu đồ loss/accuracy, cùng các chỉ số tổng hợp — dán thẳng vào báo cáo.

## 8. Danh mục tài liệu

| File | Nội dung |
| :--- | :--- |
| `project_overview.md` | Tài liệu này — tổng quan kiến trúc |
| `environment_setup.md` | Cài đặt môi trường Python 3.11 |
| `car_specs_generation.md` | Quy tắc sinh bảng thông số xe |
| `training_classifier.md` | Quy trình huấn luyện trên Colab |
| `inference_pipeline.md` | Luồng suy luận local và giao diện |
| `onnx_vs_pandas.md` | So sánh hiệu năng hai bản tư vấn |
| `cli_and_benchmark.md` | Công cụ dòng lệnh và đo hiệu năng |
| `ket_qua_danh_gia.md` | **Kết quả đánh giá hệ thống** |
| `dual_model.md` | **Hệ thống hai mô hình phân loại** |
| `frontend/README.md` | **Đặc tả frontend (Angular)** |

## 9. Trích dẫn

Krause, J., Stark, M., Deng, J., Fei-Fei, L. (2013).
*3D Object Representations for Fine-Grained Categorization.*
4th IEEE Workshop on 3D Representation and Recognition (3dRR-13),
ICCV 2013, Sydney, Australia.
