# Thu thập dữ liệu xe thị trường Việt Nam

## 1. Vấn đề

Mô hình hiện tại nhận diện 196 dòng xe từ Stanford Cars — toàn xe Mỹ/Âu
**đời ≤2012**. Với xe phổ biến tại Việt Nam (Vios, Xpander, VinFast...) hệ
thống không có lớp tương ứng.

Ví dụ thực tế: ảnh **Hyundai Accent 2025** được nhận đúng dòng xe nhưng chỉ
đạt xác suất **21.95%**, vì lớp gần nhất trong tập huấn luyện là
`Hyundai Accent Sedan 2012` — cách 13 năm và 2 thế hệ.

### Không có dataset công khai

| Nguồn | Số ảnh | Năm | Vấn đề |
| :--- | ---: | :--- | :--- |
| Stanford Cars | 16.185 | ≤2012 | Đang dùng |
| CompCars | 208.826 | ~2014 | Chỉ nghiên cứu phi thương mại, phải xin quyền |
| Kaggle | — | — | Chỉ có dữ liệu bảng (giá, doanh số), không có ảnh |

Kết luận: **phải tự thu thập**.

## 2. Giải pháp: bán tự động qua Wikimedia Commons

Wikimedia Commons chứa ảnh có **giấy phép tự do** (CC BY-SA, CC0, Public
Domain), dùng được cho đồ án mà không vướng bản quyền.

Quy trình gồm 3 bước:

```text
1. collect_vn_images.py   → tải ảnh + gán nhãn tự động từ tên file
2. review_images.py       → người duyệt lại bằng mắt, loại ảnh sai
3. prepare_vn_dataset.py  → gộp với Stanford Cars, chia train/test
```

### Vì sao vẫn cần người duyệt

Nhãn tự động từ tên file **không đủ tin cậy**. Ví dụ có thật khi thử
nghiệm: tìm `Toyota Vios XP150` trả về cả bản **sedan** lẫn **hatchback**,
mà tên file không phân biệt được — cả hai đều là
`TOYOTA VIOS XP150 China.jpg`.

Ảnh sai nhãn dạy mô hình học nhầm, và lỗi đó **rất khó phát hiện** về sau:
độ chính xác vẫn cao trên tập test (vì test cũng sai nhãn giống vậy), chỉ
hỏng khi demo với ảnh thật.

## 3. Kết quả khảo sát nguồn ảnh

Khảo sát 57 dòng xe phổ biến tại Việt Nam trên Wikimedia Commons:

| Mức | Số dòng xe |
| :--- | ---: |
| ≥200 ảnh | 48 |
| 80-199 ảnh | 4 |
| <80 ảnh | 5 |

Các dòng ít ảnh nhất đều là **VinFast** (VF 6: 43, VF 3: 71, Fadil: 23) —
hãng mới, ít ảnh trên Wikimedia.

> ⚠️ Con số tìm kiếm cao **không có nghĩa là dùng được hết**. Wikimedia
> khớp cả nội dung mô tả, nên `Ford Explorer` trả về 220.876 kết quả nhưng
> phần lớn không liên quan. Script lọc lại theo **tên file**, con số thực
> tế thấp hơn nhiều.

### Bài học khi gọi API Wikimedia

Lần khảo sát đầu dùng delay 0.3 giây giữa các lần gọi → **42/57 dòng trả về
0 kết quả**, kể cả Ford Ranger và Honda CR-V (rõ ràng phải có ảnh).

Nguyên nhân: bị giới hạn tốc độ, và API **không báo lỗi** mà trả về rỗng.
Sau khi tăng lên 1.2-3 giây kèm cơ chế thử lại, kết quả mới đúng.

> Đây là loại lỗi nguy hiểm: không có thông báo lỗi, chỉ có dữ liệu sai.
> Luôn kiểm tra lại bằng vài trường hợp mình biết chắc câu trả lời.

## 3b. VinFast — trường hợp đặc biệt

VinFast là hãng mới, ảnh trên Wikimedia rất ít. Số liệu đo ngày
09/09/2026, **sau khi lọc theo tên file**:

Số ảnh **thực tế tải được** (đã lọc theo tên file):

| Dòng xe | Kết quả tìm | Ảnh tải được | Đánh giá |
| :--- | ---: | ---: | :--- |
| VinFast VF 8 | 127 | **85** | Đủ dùng |
| VinFast VF e34 | 67 | 44 | Tạm được |
| VinFast VF 5 | 101 | 32 | Thiếu |
| VinFast Lux A2.0 | 64 | 28 | Thiếu |
| VinFast VF 3 | 71 | 26 | Thiếu |
| VinFast VF 9 | 51 | 23 | Thiếu |
| VinFast VF 7 | 53 | 23 | Thiếu |
| VinFast VF 6 | 43 | 21 | Thiếu |
| VinFast Fadil | 23 | 10 | Quá ít |

Chỉ **VF 8** đủ ảnh, VF e34 tạm được. Các dòng còn lại 10-32 ảnh — quá ít
để huấn luyện ổn định.

Giấy phép của 85 ảnh VF 8: CC BY-SA 4.0 (57), CC BY-SA 3.0 (21), CC BY 4.0
(3), CC0 (2), CC BY 2.0 (2) — **tất cả đều là giấy phép tự do**.

> **Khuyến nghị:** sau khi duyệt, loại các lớp còn **dưới 40 ảnh** khỏi tập
> huấn luyện. Lớp quá ít ảnh làm mô hình học không ổn định và kéo tụt độ
> chính xác của cả tập.

Danh mục vẫn giữ đủ 10 dòng VinFast để anh tự quyết sau khi thấy số liệu
thật. Nếu muốn giữ tất cả, cần bổ sung ảnh từ nguồn khác (chụp thực tế,
hoặc xin phép dùng ảnh từ trang chủ VinFast).

## 4. Danh mục xe

`data/vn_car_classes.json` — **50 dòng xe** thuộc 14 hãng, kèm thông số:

| Trường | Nội dung |
| :--- | :--- |
| `class_name` | Tên lớp, cùng định dạng Stanford Cars |
| `search` | Từ khóa tìm kiếm trên Wikimedia |
| `brand`, `model`, `body_style`, `year`, `seats` | Thông số cơ bản |
| `price_million_vnd` | Giá niêm yết tại VN (2024-2025), làm tròn |
| `fuel_l_per_100km` | Mức tiêu hao công bố |

> Khác với `car_specs.csv` (số liệu **mô phỏng**), giá và mức tiêu hao ở
> đây lấy theo **giá niêm yết thực tế** của hãng tại Việt Nam.

Xe điện VinFast có `fuel_l_per_100km = 0.0` — cần lưu ý khi lọc theo mức
tiêu hao.

## 5. Cách chạy

### Bước 1 — Thu thập ảnh

```powershell
# Thử một dòng xe trước
.\.venv\Scripts\python.exe scripts/collect_vn_images.py --only "Toyota Vios" --limit 30

# Chạy toàn bộ (mất khoảng 1-2 giờ)
.\.venv\Scripts\python.exe scripts/collect_vn_images.py --limit 200
```

Ảnh lưu vào `data/raw/vn_cars/<tên lớp>/`, kèm file `_credits.json` ghi
nguồn và giấy phép từng ảnh — dùng để trích dẫn trong báo cáo.

### Bước 2 — Duyệt ảnh

```powershell
.\.venv\Scripts\python.exe scripts/review_images.py
```

Mở <http://localhost:8600>. Giao diện hiện lưới ảnh của từng lớp:

- **Bấm vào ảnh sai** để đánh dấu loại (viền đỏ)
- Bấm **"Lưu và sang lớp tiếp theo"**
- Ảnh bị loại chuyển sang `_rejected/`, **không xóa hẳn**

Ước tính: **3-5 giờ** cho toàn bộ 42 dòng xe.

### Bước 3 — Chuẩn bị dataset

*(chưa triển khai — sẽ gộp với Stanford Cars và chia train/test)*

## 6. Ghi công nguồn ảnh

Giấy phép CC BY-SA yêu cầu **ghi tên tác giả**. File `_credits.json` trong
mỗi thư mục lớp lưu sẵn:

```json
{
  "title": "File:TOYOTA VIOS XP150 China.jpg",
  "license": "CC BY-SA 4.0",
  "artist": "Tên tác giả",
  "descriptionurl": "https://commons.wikimedia.org/wiki/File:..."
}
```

Trong báo cáo cần nêu: *"Ảnh xe thị trường Việt Nam thu thập từ Wikimedia
Commons, giấy phép CC BY-SA 4.0. Danh sách tác giả đầy đủ trong file
`_credits.json` của từng lớp."*

## 7. Hạn chế đã biết

**Ảnh là xe thị trường quốc tế.** Wikimedia chủ yếu có ảnh chụp tại Trung
Quốc, Thái Lan, Indonesia. Bản xe tại Việt Nam có thể khác đôi chút về
ngoại hình (đèn, lưới tản nhiệt, mâm xe).

**Số lượng không đều.** Một số dòng chỉ có vài chục ảnh sau khi lọc, ít hơn
nhiều so với mức ~42 ảnh/lớp của Stanford Cars. Cần cân nhắc loại các lớp
quá ít ảnh khỏi tập huấn luyện.

**VinFast thiếu ảnh nghiêm trọng.** Fadil chỉ 23 kết quả tìm kiếm, VF 6 có
43 — nhiều khả năng không đủ để huấn luyện.

**Chưa có ảnh chụp trong điều kiện thực tế Việt Nam.** Ảnh Wikimedia phần
lớn chụp tại triển lãm hoặc showroom, ánh sáng tốt. Ảnh người dùng chụp
ngoài đường sẽ khó hơn.
