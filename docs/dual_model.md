# Hệ thống hai mô hình phân loại

## 1. Vì sao hai mô hình

Dự án có hai mô hình phân loại chạy song song:

| Mô hình | Lớp | Dữ liệu | Phạm vi |
| :--- | ---: | :--- | :--- |
| `car_classifier.onnx` | 196 | Stanford Cars | Xe quốc tế, đời ≤2012 |
| `vn_car_classifier.onnx` | 20 | Wikimedia Commons | Xe thị trường VN, 2021-2024 |

Hai tập xe **gần như không giao nhau** — không thể gộp thành một mô hình
mà không train lại từ đầu, và gộp sẽ làm mất cân bằng lớp nặng hơn.

## 2. Cách chọn kết quả

Pipeline chạy **cả hai mô hình** trên cùng ảnh đã crop, rồi chọn mô hình
tự tin hơn.

### Vấn đề: so sánh không công bằng

Mô hình 20 lớp dễ đạt xác suất cao hơn mô hình 196 lớp. Đoán mò ở 20 lớp
cho 5%, ở 196 lớp chỉ 0.5%. So sánh trực tiếp sẽ **luôn thiên vị mô hình
VN**.

### Giải pháp: hệ số phạt

```python
VN_CONFIDENCE_PENALTY = 0.40

vn_score = vn_predictions[0].confidence * VN_CONFIDENCE_PENALTY
if vn_score > international[0].confidence:
    chọn mô hình VN
```

### Chọn hệ số bằng đo đạc

Đo trên **326 ảnh xe VN** (tập test) + **18 ảnh xe quốc tế** (Wikimedia,
các dòng có trong Stanford Cars):

| Hệ số | VN top-1 | QT top-1 | Trung bình |
| ---: | ---: | ---: | ---: |
| 0.20 | 32.5% | 55.6% | 44.0% |
| 0.30 | 47.5% | 55.6% | 51.6% |
| **0.40** | **54.6%** | **55.6%** | **55.1%** |
| 0.50 | 60.1% | 50.0% | 55.1% |
| 0.75 | 67.8% | 44.4% | 56.1% |
| 1.00 | 70.9% | 38.9% | 54.9% |

**Không có hệ số nào tốt cho cả hai chiều** — đây là đánh đổi thật, không
phải vấn đề tinh chỉnh. Chọn **0.40** vì hai chiều gần bằng nhau (54.6% và
55.6%). Từ 0.50 trở lên, xe quốc tế tụt nhanh.

> ⚠️ Tập test xe quốc tế chỉ có **18 ảnh** — quá nhỏ để kết luận chắc
> chắn. Xu hướng rõ nhưng con số cụ thể có sai số lớn. Nếu có thời gian
> nên tải thêm ~100 ảnh rồi đo lại.

### Vì sao không gộp chung danh sách

Đã thử gộp (trộn top-k của hai mô hình rồi xếp hạng chung) và đo:

| Cách làm | QT top-1 | VN top-1 |
| :--- | ---: | ---: |
| Chỉ mô hình quốc tế | **61.1%** | — |
| Gộp, phạt 0.75 | 44.4% | 67.8% |
| Gộp, phạt 0.40 | 55.6% | 54.6% |
| **Chọn một mô hình, phạt 0.40** | **55.6%** | **54.6%** |

Gộp luôn làm xe quốc tế tệ đi vì các lớp xe VN vô nghĩa chèn vào top-5 —
ở hệ số 0.75, **67.8% ô trong top-5 của ảnh xe quốc tế bị xe VN chiếm**.

Ví dụ thật: ảnh 4 xe thể thao (Alfa Romeo, Audi RS5, BMW M4, Mercedes)
khi gộp cho ra Hyundai Accent, Toyota Yaris, Toyota Vios, Ferrari 458 —
**3/4 là xe Việt Nam**, sai hoàn toàn.

### Ví dụ hoạt động đúng

| Ảnh | Quốc tế | Việt Nam | Sau phạt | Chọn |
| :--- | ---: | ---: | ---: | :--- |
| VinFast VF 8 | 13.39% (Bugatti) | 42.64% (VF 8) | 17.06% | **VN** ✅ |
| Hyundai Accent 2024 | 90.04% (Accent 2012) | 16.67% (Yaris) | 6.67% | **Quốc tế** ✅ |

Trường hợp Accent đáng chú ý: mô hình quốc tế **vừa tự tin hơn vừa đúng**
(đúng hãng, đúng dòng, chỉ khác đời).

## 3. Kết quả trong response

Mỗi kết quả nhận diện mang thêm hai trường:

| Trường | Ý nghĩa |
| :--- | :--- |
| `source` | `"international"` hoặc `"vietnam"` |
| `alternative` | Dự đoán của mô hình còn lại, để đối chiếu |

Giao diện **không hiển thị** nguồn mô hình — người dùng chỉ cần kết quả,
không cần biết mô hình nào đưa ra. Hai trường này dùng cho thống kê, gỡ
lỗi, và API (client có thể tự quyết cách hiển thị).

## 4. Mô hình VN là tùy chọn

Thiếu file `models/vn_car_classifier.onnx` thì hệ thống **vẫn chạy bình
thường** với mô hình quốc tế. Log ghi:

```text
Khong co mo hinh xe Viet Nam, chi dung mo hinh quoc te.
```

Tương tự với `models/vn_car_specs.json` — module tư vấn chỉ có 196 dòng xe
quốc tế thay vì 216.

## 5. Kết quả mô hình xe Việt Nam

Đo trên tập test 326 ảnh (20 lớp):

| Chỉ số | Kết quả |
| :--- | ---: |
| Top-1 | **75.15%** |
| Top-5 | **92.02%** |
| Balanced accuracy | **71.59%** |

**Mô hình dự đoán đủ cả 20 lớp**, không bỏ qua lớp nào — các kỹ thuật cân
bằng đã có tác dụng. Balanced (71.59%) gần sát top-1 (75.15%) chứng tỏ
không có lớp nào "chết".

### Độ chính xác theo lớp

| Lớp | Tỷ lệ | Ảnh train |
| :--- | ---: | ---: |
| Hyundai Creta | 100% | 42 |
| VinFast VF 6 | 100% | 17 |
| VinFast Fadil | 100% | 11 |
| Hyundai Accent | 96.8% | 126 |
| Toyota Corolla Cross | 86.5% | 150 |
| … | | |
| VinFast Lux A2.0 | 42.9% | 27 |
| VinFast VF 8 | 41.2% | 68 |
| VinFast VF 3 | 40.0% | 22 |
| VinFast VF 5 | 33.3% | 26 |
| VinFast Lux SA2.0 | 33.3% | 14 |

> Fadil và VF 6 đạt 100% nhưng tập test chỉ có 1-4 ảnh — con số này **không
> đáng tin về mặt thống kê**. Cần nêu rõ khi báo cáo.

### Phân tích lỗi

Trong 81 lỗi:

- **54 lỗi (67%) là nhầm giữa các dòng CÙNG HÃNG**
- 27 lỗi (33%) nhầm khác hãng

Các cặp nhầm nhiều nhất:

| Thực tế | Dự đoán | Lần |
| :--- | :--- | ---: |
| VinFast VF 8 | VinFast VF 9 | 3 |
| VinFast VF 8 | VinFast VF 6 | 3 |
| VinFast Lux A2.0 | VinFast Lux SA2.0 | 3 |
| Toyota Vios | Toyota Camry | 3 |
| Toyota Yaris | Toyota Vios | 3 |

Đây là kiểu lỗi **dễ hiểu**: các dòng SUV điện VinFast có ngôn ngữ thiết kế
rất giống nhau; Lux A2.0 và SA2.0 cùng nền tảng chỉ khác sedan/SUV; Vios và
Yaris cũng cùng nền tảng.

Nghĩa là mô hình **nhận đúng hãng và phong cách thiết kế**, chưa phân biệt
được dòng cụ thể trong cùng họ.

## 6. Hạn chế

**Nhãn chưa duyệt bằng mắt.** Ảnh gán nhãn tự động từ tên file Wikimedia,
chỉ lọc được ảnh **có năm trong tên**. Độ chính xác báo cáo **cao hơn thực
tế** vì tập test sai nhãn giống tập train.

**Tập test quá nhỏ với lớp ít ảnh.** Fadil chỉ 1 ảnh test, VF 6 có 4 ảnh —
kết quả 100% không có ý nghĩa thống kê.

**Chỉ 20/50 lớp đã thu thập.** Còn thiếu Mitsubishi, Ford, Mazda, Kia,
Suzuki, Nissan, Subaru, Isuzu, MG và 3/4 dòng Honda.

**Ba lớp không xác định được đời xe.** Bộ lọc theo năm chỉ bắt được ảnh có
năm dạng 4 chữ số trong tên file. Wikimedia còn dùng dạng viết tắt
(`'00-'02`, `03-'06`) và rất nhiều ảnh không ghi năm.

Hậu quả nặng nhất ở lớp `Hyundai Accent Sedan 2024`: chỉ **5% ảnh đúng
đời**, còn lại là Accent 2006-2014 (phần lớn là hatchback). Mô hình vì thế
không nhận ra Accent 2024 thật.

Ba lớp đã được **đổi tên bỏ năm** cho trung thực với dữ liệu:

| Tên cũ | Tên mới | Ảnh đúng đời |
| :--- | :--- | ---: |
| Hyundai Accent Sedan 2024 | `Hyundai Accent Sedan` | 5% |
| Toyota Vios Sedan 2023 | `Toyota Vios Sedan` | 67% |
| Honda City Sedan 2023 | `Honda City Sedan` | 69% |

17 lớp còn lại đạt 80-100% đúng đời nên giữ nguyên tên.

> Đổi tên **không cần train lại** — chỉ đổi nhãn hiển thị, thứ tự lớp giữ
> nguyên. Chạy `scripts/rename_mixed_year_classes.py` để áp dụng.

**Tập test xe quốc tế quá nhỏ.** Chỉ 18 ảnh — hệ số 0.40 chọn dựa trên số
liệu này nên có sai số lớn. Nên tải thêm ~100 ảnh xe có trong Stanford Cars
rồi đo lại.

**Không có hệ số nào tốt cho cả hai chiều.** Đây là hạn chế cố hữu của việc
chạy hai mô hình độc lập trên cùng một ảnh. Cách giải quyết triệt để là
train một mô hình duy nhất phủ cả hai tập xe — nhưng phải train lại từ đầu
và mất cân bằng lớp sẽ nặng hơn.
