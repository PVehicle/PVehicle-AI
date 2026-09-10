# Huấn luyện mô hình nhận diện xe Việt Nam

## 1. Vì sao là mô hình riêng

Dự án có **hai mô hình phân loại độc lập**:

| Mô hình | Số lớp | Dữ liệu | Dùng cho |
| :--- | ---: | :--- | :--- |
| `car_classifier.onnx` | 196 | Stanford Cars | Xe quốc tế, đời ≤2012 |
| `vn_car_classifier.onnx` | 20 | Wikimedia Commons | Xe thị trường VN |

Lý do tách riêng thay vì gộp thành một mô hình 216 lớp:

- **Không phá mô hình đang chạy tốt.** Mô hình Stanford Cars đạt 83.02%
  top-1, gộp lại phải train lại từ đầu và có thể kém hơn.
- **Mất cân bằng sẽ nặng hơn.** Stanford Cars có ~42 ảnh/lớp, xe VN có
  14-187 ảnh/lớp. Gộp lại làm chênh lệch càng lớn.
- **Dễ đánh giá hơn.** Đo được riêng từng mô hình, biết cái nào tốt cái nào
  kém.

Ứng dụng chạy cả hai rồi so sánh độ tin cậy để chọn kết quả.

## 2. Vấn đề lớn nhất: mất cân bằng lớp

| Lớp | Ảnh |
| :--- | ---: |
| Toyota Corolla Cross | 187 |
| Toyota Camry | 186 |
| VinFast VF 8 | 85 |
| VinFast Lux SA2.0 | 17 |
| **VinFast Fadil** | **14** |

Chênh lệch **13.4 lần**.

### Hậu quả nếu không xử lý

Mô hình học được rằng "đoán Fadil gần như luôn sai" nên nó **tránh dự đoán
lớp đó**. Độ chính xác tổng thể vẫn cao (các lớp lớn đúng nhiều) nhưng
Fadil gần **0%**.

### Ba lớp bảo vệ

**1. Lấy mẫu cân bằng** (`WeightedRandomSampler`) — mỗi epoch, ảnh của lớp
ít được lấy nhiều lần hơn để các lớp xuất hiện đều nhau.

**2. Trọng số lớp trong hàm mất mát** — lớp ít ảnh được nhân trọng số cao:

```text
trọng_số = 100 / số_ảnh_của_lớp     (giới hạn [1, 8])
```

Fadil (14 ảnh) → **7.14**; Corolla Cross (187 ảnh) → **1.00**.

**3. Tăng cường dữ liệu mạnh hơn** — `RandomResizedCrop(scale=0.6-1.0)`,
xoay ±10°, `ColorJitter` mạnh, `RandomErasing(p=0.3)`.

### Chọn mô hình tốt nhất theo balanced accuracy

Notebook **không** chọn checkpoint theo top-1 mà theo **balanced accuracy**
(trung bình độ chính xác từng lớp).

Lý do: với tập mất cân bằng, top-1 cao có thể che giấu việc vài lớp không
hoạt động. Balanced accuracy phạt đúng trường hợp đó.

## 3. Làm sạch dữ liệu: lọc theo năm

### Vấn đề phát hiện được

Tìm `Toyota Vios` trên Wikimedia trả về ảnh từ **2003 đến 2025** — bốn thế
hệ xe khác hẳn nhau. Gán tất cả vào lớp `Toyota Vios Sedan 2023` dạy mô
hình rằng chúng là một.

Mức độ trước khi lọc:

| Lớp | Ảnh đời cũ | Khoảng năm |
| :--- | ---: | :--- |
| Toyota Innova 2023 | 88 | 2004-2024 |
| Toyota Fortuner 2024 | 85 | 2006-2025 |
| Toyota Vios 2023 | 50 | 2003-2025 |
| Hyundai Accent 2024 | 42 | 2000-2022 |

### Cách lọc

May mắn là tên file Wikimedia thường **có năm sản xuất** ở đầu
(`2011 Toyota Vios J front.jpg`), nên lọc được tự động:

```powershell
.\.venv\Scripts\python.exe scripts/filter_by_year.py --dry-run
.\.venv\Scripts\python.exe scripts/filter_by_year.py
```

Quy tắc:

| Trường hợp | Xử lý |
| :--- | :--- |
| Có năm, lệch quá 3 năm so với nhãn | **Loại** |
| Có năm, lệch ≤ 3 năm | Giữ |
| Không có năm trong tên | **Giữ** (không đoán bừa) |
| Tên chứa "rim of", "wheel of"... | **Loại** (chỉ chụp bộ phận) |

> **Vì sao dung sai 3 năm:** một thế hệ xe thường kéo dài 5-7 năm, và bản
> facelift giữa vòng đời vẫn rất giống bản gốc. Lệch 3 năm thường vẫn cùng
> thế hệ.

Ảnh bị loại chuyển sang `_year_rejected/`, không xóa hẳn.

### Kết quả

Loại **371 ảnh (18.2%)**, còn **1.673 ảnh**. Các lớp VinFast không mất ảnh
nào — hãng mới, không có đời cũ.

## 4. Quy trình đầy đủ

```powershell
# 1. Thu thập ảnh
.\.venv\Scripts\python.exe scripts/collect_vn_images.py --limit 200

# 2. Lọc ảnh đời cũ
.\.venv\Scripts\python.exe scripts/filter_by_year.py

# 3. (Tùy chọn) Duyệt bằng mắt — chính xác nhất
.\.venv\Scripts\python.exe scripts/review_images.py

# 4. Chia train/test, tính trọng số lớp
.\.venv\Scripts\python.exe scripts/prepare_vn_dataset.py

# 5. Đóng gói để tải lên Colab
.\.venv\Scripts\python.exe scripts/pack_vn_dataset.py
```

Bước 5 tự thu nhỏ ảnh về 448px trước khi nén — ảnh gốc Wikimedia rất lớn
trong khi mô hình chỉ dùng 224px. **Giảm 78%**: 300 MB → 62 MB.

## 5. Chạy notebook

`notebooks/train_vn_cars_colab.ipynb` trên Google Colab (T4 GPU).

Tải lên **hai file**:

| File | Nguồn |
| :--- | :--- |
| `vn_cars.zip` | Thư mục gốc dự án (bước 5) |
| `vn_dataset_config.json` | `data/` |

Cấu hình huấn luyện:

| Tham số | Giá trị | Khác gì Stanford Cars |
| :--- | :--- | :--- |
| Epoch | 30 | Nhiều hơn (20) vì ít dữ liệu |
| Learning rate | 3e-4 | Thấp hơn (1e-3) |
| Weight decay | 1e-3 | Cao hơn (1e-4) |
| Dropout | 0.4 | Cao hơn mặc định (0.2) |
| Batch size | 32 | Nhỏ hơn (64) |

Các thay đổi đều nhằm **chống quá khớp** — tập chỉ 1.673 ảnh so với 16.185.

Thời gian: **~20-30 phút** trên T4.

## 6. Đọc kết quả

Notebook báo cáo ba chỉ số:

| Chỉ số | Ý nghĩa |
| :--- | :--- |
| Top-1 | Tỷ lệ đoán đúng ngay lựa chọn đầu |
| Top-5 | Tỷ lệ xe đúng nằm trong 5 gợi ý |
| **Balanced** | **Trung bình độ chính xác từng lớp** |

**Balanced accuracy là chỉ số đáng tin nhất** với tập mất cân bằng. Nếu
top-1 cao mà balanced thấp, mô hình đang bỏ qua các lớp ít ảnh.

Notebook cũng in **bảng độ chính xác theo từng lớp** và **ma trận nhầm
lẫn** — hai thứ này cần đưa vào báo cáo, không chỉ con số tổng thể.

## 7. Hạn chế cần nêu trong báo cáo

**Nhãn chưa được duyệt bằng mắt.** Ảnh gán nhãn tự động từ tên file, lọc
theo năm chỉ xử lý được ảnh **có năm trong tên**. Vẫn còn ảnh sai nhãn —
ví dụ Vios sedan và hatchback có tên file giống hệt nhau.

Hệ quả: **độ chính xác báo cáo sẽ cao hơn thực tế**, vì tập test cũng sai
nhãn giống tập train nên lỗi tự triệt tiêu.

**Lớp ít ảnh vẫn kém.** Các kỹ thuật cân bằng chỉ giảm nhẹ chứ không xóa bỏ
được vấn đề. Fadil 14 ảnh vẫn sẽ kém hơn Camry 186 ảnh.

**Ảnh chủ yếu là xe thị trường quốc tế.** Wikimedia có nhiều ảnh chụp tại
Trung Quốc, Thái Lan, Indonesia. Bản xe tại Việt Nam có thể khác về đèn,
lưới tản nhiệt, mâm xe.

**Chỉ 20/50 lớp đã thu thập.** Còn thiếu Mitsubishi, Ford, Mazda, Kia,
Suzuki, Nissan, Subaru, Isuzu, MG và 3/4 dòng Honda.
