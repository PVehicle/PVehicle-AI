# PVehicle-AI 🚗🤖

**PVehicle-AI** là hệ thống Trí tuệ Nhân tạo đa nhiệm, tích hợp khả năng
nhận diện dòng xe ô tô qua hình ảnh và tư vấn dòng xe phù hợp với nhu cầu
cá nhân hóa.

Toàn bộ khâu suy luận chạy tại **môi trường cục bộ bằng ONNX Runtime trên
CPU** — không cần GPU, không cần cài PyTorch.

---

## 📈 Kết quả

Hệ thống có **hai mô hình phân loại** chạy song song:

| Mô hình | Lớp | Top-1 | Top-5 |
| :--- | ---: | ---: | ---: |
| Xe quốc tế (Stanford Cars) | 196 | **83.02%** | **95.26%** |
| Xe thị trường Việt Nam | 20 | **75.15%** | **92.02%** |

| Chỉ số chung | Kết quả |
| :--- | ---: |
| Tốc độ suy luận (CPU) | 135 ms/ảnh — 7.4 ảnh/giây |
| Kiểm thử tự động | 96/96 pass |

Chi tiết: `docs/ket_qua_danh_gia.md`, `docs/dual_model.md`.

---

## 🌟 Tính năng chính

- **🔍 Nhận diện dòng xe** — kiến trúc 2 tầng: YOLOv8 khoanh vùng xe trong
  ảnh, sau đó EfficientNet-B0 phân loại dòng xe.
- **🌍🇻🇳 Hai mô hình song song** — xe quốc tế (196 dòng, đời ≤2012) và xe
  thị trường Việt Nam (20 dòng, 2021-2024); hệ thống tự chọn kết quả đáng
  tin hơn.
- **💡 Tư vấn thông minh** — gợi ý xe theo ngân sách, số chỗ ngồi, kiểu
  dáng và mức tiêu hao nhiên liệu.
- **🔗 Liên kết hai module** — nhận diện ra xe nào thì gợi ý ngay các xe
  tương tự cùng tầm giá.
- **⚡ Suy luận nhanh trên CPU** — ONNX Runtime, độc lập hoàn toàn với
  framework huấn luyện gốc.
- **🖥️ Giao diện web** — Streamlit, hỗ trợ tải ảnh lên hoặc dán URL.

---

## 🏗️ Kiến trúc hệ thống

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

| Tầng | Mô hình | Huấn luyện |
| :--- | :--- | :--- |
| 1. Phát hiện xe | YOLOv8n pretrained COCO | Không cần train |
| 2a. Phân loại xe quốc tế | EfficientNet-B0 (196 lớp) | Trên Google Colab |
| 2b. Phân loại xe Việt Nam | EfficientNet-B0 (20 lớp) | Trên Google Colab |

Tầng 2 chạy **cả hai mô hình** rồi chọn kết quả đáng tin hơn — hai tập xe
gần như không giao nhau (quốc tế đời ≤2012, Việt Nam đời 2021-2024).
Chi tiết: `docs/dual_model.md`.

> **Vì sao tách 2 tầng?** Stanford Cars không có bounding box theo từng dòng
> xe, nên không thể train YOLOv8 ra thẳng 196 lớp. Dùng YOLOv8 pretrained
> để khoanh vùng rồi train riêng mô hình phân loại vừa đúng dữ liệu, vừa
> tiết kiệm rất nhiều thời gian.

---

## 🛠️ Công nghệ sử dụng

| Hạng mục | Công nghệ |
| :--- | :--- |
| Ngôn ngữ | Python **3.11** (bắt buộc, xem lưu ý bên dưới) |
| Thị giác máy tính | YOLOv8 (Ultralytics), OpenCV |
| Học máy | PyTorch (chỉ khi huấn luyện), Scikit-learn, Pandas |
| Tối ưu & triển khai | ONNX, ONNX Runtime |
| Giao diện | Streamlit |
| REST API | FastAPI, Uvicorn, Docker |

> ⚠️ **Bắt buộc Python 3.11.** `onnxruntime`, `torch` và `opencv-python`
> chưa có wheel cho Python 3.13+. Chi tiết: `docs/environment_setup.md`.

---

## 📂 Cấu trúc thư mục

```text
PVehicle-AI/
│
├── app.py                          # Giao diện Streamlit
├── requirements.txt                # Thư viện để CHẠY (máy local)
├── requirements-train.txt          # Thư viện để HUẤN LUYỆN (Colab)
│
├── data/
│   ├── stanford_cars_classes.txt   # 196 tên lớp gốc
│   └── car_specs.csv               # Bảng thông số xe (sinh tự động)
│
├── models/                         # Mô hình ONNX (không đẩy lên git)
│   ├── yolov8n.onnx                # Tầng 1: phát hiện xe
│   ├── car_classifier.onnx         # Tầng 2: phân loại (196 lớp quốc tế)
│   ├── class_names.json            # 196 nhãn theo đúng thứ tự
│   ├── vn_car_classifier.onnx      # Tầng 2: phân loại xe VN (20 lớp)
│   ├── vn_class_names.json         # 20 nhãn xe Việt Nam
│   └── vn_car_specs.json           # Thông số xe VN (giá niêm yết thật)
│
├── src/
│   ├── utils.py                    # Logging, định vị đường dẫn
│   ├── cv/
│   │   ├── detector.py             # Tầng 1: YOLOv8
│   │   ├── classifier.py           # Tầng 2: phân loại
│   │   └── pipeline.py             # Ghép hai tầng
│   ├── rec/
│   │   ├── recommender.py          # Module tư vấn (bản dùng chính)
│   │   └── onnx_recommender.py     # Bản ONNX, để so sánh hiệu năng
│   └── api/                        # REST API (FastAPI)
│       ├── main.py                 # Khởi tạo app, middleware
│       ├── config.py               # Cấu hình qua biến môi trường
│       ├── schemas.py              # Kiểu dữ liệu request/response
│       ├── dependencies.py         # Quản lý vòng đời mô hình
│       ├── security.py             # Xác thực API key
│       └── routers/                # Các nhóm endpoint
│
├── scripts/
│   ├── generate_car_specs.py       # Sinh bảng thông số xe
│   ├── build_train_notebook.py     # Sinh notebook huấn luyện
│   ├── make_dummy_models.py        # Mô hình giả để kiểm thử
│   ├── recognize.py                # Nhận diện từ dòng lệnh
│   ├── benchmark_inference.py      # Đo hiệu năng suy luận
│   ├── evaluate_model.py           # Sinh báo cáo từ checkpoint
│   └── export_recommender_onnx.py  # Export + so sánh module tư vấn
│
├── notebooks/
│   └── train_classifier_colab.ipynb  # Huấn luyện trên Colab
│
├── tests/                          # 96 unit test
└── docs/                           # Tài liệu kỹ thuật
```

---

## 🚀 Cài đặt

### 1. Tạo môi trường ảo

```powershell
py -3.11 -m venv .venv
```

> Dùng `py -3.11` thay cho `python` — lệnh `python` trên Windows thường trỏ
> tới shortcut của Microsoft Store (phiên bản khác, bị giới hạn quyền ghi).

### 2. Cài thư viện

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Máy local **không cần** cài `torch` (~2,5 GB) — việc huấn luyện diễn ra trên
Google Colab.

### 3. Sinh bảng thông số xe

```powershell
.\.venv\Scripts\python.exe scripts/generate_car_specs.py
```

### 4. Chuẩn bị mô hình

**Cách A — dùng mô hình thật:** huấn luyện bằng
`notebooks/train_classifier_colab.ipynb` trên Google Colab (~1 giờ với GPU
T4), rồi chép 3 file kết quả vào `models/`.

**Cách B — chạy thử trước bằng mô hình giả:**

```powershell
.\.venv\Scripts\python.exe scripts/make_dummy_models.py
```

Mô hình giả có đúng cấu trúc nhưng trọng số ngẫu nhiên, nên **kết quả nhận
diện vô nghĩa**. Chỉ dùng để kiểm tra code chạy thông.

---

## ▶️ Chạy ứng dụng

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Mở tại <http://localhost:8501>.

Giao diện có 2 tab:

- **Nhận diện xe** — tải ảnh lên hoặc dán URL, xem top-5 dự đoán, thông số
  kỹ thuật và các xe tương tự.
- **Tư vấn xe** — nhập nhu cầu qua form, nhận danh sách xe phù hợp.

> Tab **Tư vấn xe** chạy được ngay cả khi chưa có mô hình, vì nó chỉ cần
> `data/car_specs.csv`.

### Chạy từ dòng lệnh

Không cần mở giao diện:

```powershell
# Nhận diện một ảnh, kèm gợi ý xe tương tự
.\.venv\Scripts\python.exe scripts/recognize.py anh_xe.jpg --recommend

# Xử lý cả thư mục
.\.venv\Scripts\python.exe scripts/recognize.py thu_muc_anh/

# Đo hiệu năng suy luận
.\.venv\Scripts\python.exe scripts/benchmark_inference.py
```

Chi tiết: `docs/cli_and_benchmark.md`.

### Chạy REST API

Phục vụ ứng dụng khác (mobile, web, hệ thống bên thứ ba):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-api.txt
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

Tài liệu tương tác: <http://localhost:8000/docs>

Hoặc chạy bằng Docker:

```bash
cp .env.example .env
docker compose up --build
```

Chi tiết: `docs/rest_api.md`.

---

## 🧪 Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

Các test cần mô hình sẽ tự bỏ qua nếu `models/` còn trống.

---

## 📊 Dữ liệu

**Ảnh:** [Stanford Cars](https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars)
— 16.185 ảnh, 196 lớp (hãng + dòng + kiểu dáng + năm).

> Trang gốc của Stanford đã bị gỡ bỏ, dự án dùng bản mirror trên Kaggle giữ
> nguyên file `.mat` (có bounding box).

**Thông số xe:** Stanford Cars không cung cấp giá bán hay mức tiêu hao
nhiên liệu, nên `data/car_specs.csv` được sinh tự động ở hai mức:

| Mức | Trường | Độ tin cậy |
| :--- | :--- | :--- |
| Suy luận | hãng, dòng, kiểu dáng, năm, số chỗ | **Chính xác** |
| Mô phỏng | giá bán, mức tiêu hao | **Số liệu mô phỏng** |

Chi tiết quy tắc sinh: `docs/car_specs_generation.md`.

---

## 📚 Tài liệu

| File | Nội dung |
| :--- | :--- |
| `docs/project_overview.md` | Tổng quan dự án |
| `docs/environment_setup.md` | Cài đặt môi trường |
| `docs/car_specs_generation.md` | Quy tắc sinh bảng thông số xe |
| `docs/training_classifier.md` | Quy trình huấn luyện |
| `docs/inference_pipeline.md` | Luồng suy luận và giao diện |
| `docs/cli_and_benchmark.md` | Công cụ dòng lệnh và đo hiệu năng |
| `docs/onnx_vs_pandas.md` | So sánh hiệu năng hai bản tư vấn |
| `docs/ket_qua_danh_gia.md` | **Kết quả đánh giá hệ thống** |
| `docs/dual_model.md` | **Hệ thống hai mô hình phân loại** |

---

## 📝 Ghi chú

- Bộ dữ liệu Stanford Cars gồm xe Mỹ/Âu đời ≤2012, **không phải xe thị
  trường Việt Nam**.
- Giá bán và mức tiêu hao trong `car_specs.csv` là **số liệu mô phỏng**,
  không phải giá thị trường thật.

---

## 📖 Trích dẫn

Krause et al., *3D Object Representations for Fine-Grained Categorization*,
4th IEEE Workshop on 3D Representation and Recognition (3dRR-13), ICCV 2013.
