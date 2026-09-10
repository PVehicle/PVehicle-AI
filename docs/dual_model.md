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
VN_CONFIDENCE_PENALTY = 0.75

vn_score = vn_predictions[0].confidence * VN_CONFIDENCE_PENALTY
if vn_score > international[0].confidence:
    chọn mô hình VN
```

Giá trị 0.75 chọn theo kinh nghiệm: đủ bù chênh lệch số lớp, nhưng không
làm mất ưu thế của mô hình VN khi nó thực sự nhận ra xe.

### Ví dụ thực tế

| Ảnh | Quốc tế | Việt Nam | Sau phạt | Chọn |
| :--- | ---: | ---: | ---: | :--- |
| VinFast VF 8 | 13.39% (Bugatti) | 42.64% (VF 8) | 31.98% | **VN** ✅ |
| Toyota Vios | 22.68% (Kizashi) | 32.13% (Innova) | 24.09% | **VN** |
| Hyundai Accent 2024 | 90.04% (Accent 2012) | 16.67% (Yaris) | 12.50% | **Quốc tế** ✅ |

Trường hợp Accent đáng chú ý: mô hình quốc tế **vừa tự tin hơn vừa đúng**
(đúng hãng, đúng dòng, chỉ khác đời). Cơ chế chọn hoạt động đúng.

## 3. Kết quả trong response

Mỗi kết quả nhận diện mang thêm hai trường:

| Trường | Ý nghĩa |
| :--- | :--- |
| `source` | `"international"` hoặc `"vietnam"` |
| `alternative` | Dự đoán của mô hình còn lại, để đối chiếu |

Giao diện hiển thị nhãn 🌍 hoặc 🇻🇳, kèm mục có thể mở ra xem mô hình kia
đoán gì. CLI in `[quoc te]` hoặc `[xe VN]`. API trả trường `source`.

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

**Hệ số phạt 0.75 chọn theo kinh nghiệm**, chưa tối ưu bằng thực nghiệm có
hệ thống. Nếu có thời gian nên quét thử nhiều giá trị trên một tập ảnh đại
diện.
