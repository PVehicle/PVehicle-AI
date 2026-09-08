# Sinh bảng thông số xe (`car_specs.csv`)

## 1. Vấn đề

Module tư vấn cần bảng thông số xe (giá bán, phân khúc, số chỗ, mức tiêu hao
nhiên liệu). Nhưng dataset **Stanford Cars chỉ cung cấp ảnh và tên lớp** —
không có bất kỳ thông số kỹ thuật nào.

196 lớp này là xe Mỹ/Âu đời ≤2012, không có nguồn CSV công khai nào tổng hợp
sẵn giá bán và mức tiêu hao. Tra tay đủ 196 dòng × 5 trường là gần 1000 ô,
tốn hàng chục giờ mà không đóng góp gì cho phần AI của đồ án.

**Giải pháp:** sinh bảng thông số bằng script, theo quy tắc xác định
(deterministic), tách bạch rõ dữ liệu chính xác và dữ liệu mô phỏng.

## 2. Hai mức độ tin cậy của dữ liệu

| Mức | Các trường | Cách có được | Độ tin cậy |
| :--- | :--- | :--- | :--- |
| `derived` | `brand`, `model`, `body_style`, `year`, `seats` | Suy trực tiếp từ tên lớp | **Chính xác** |
| `simulated` | `price_million_vnd`, `fuel_l_per_100km` | Sinh theo công thức | **Mô phỏng** |

Tên lớp Stanford Cars có cấu trúc `<Hãng> <Dòng> <Kiểu dáng> <Năm>`, ví dụ
`Acura RL Sedan 2012`. Nhờ vậy hãng, dòng, kiểu dáng, năm và số chỗ ngồi đều
**tách ra được chính xác**, không phải đoán.

Chỉ có **giá bán** và **mức tiêu hao** là mô phỏng. Cột `data_source` trong
CSV ghi rõ điều này cho từng dòng.

> **Lưu ý khi báo cáo:** phải nêu rõ giá bán và mức tiêu hao là số liệu mô
> phỏng, không phải giá thị trường thật.

## 3. Quy tắc suy luận (derived)

### 3.1. Tách hãng xe

Lấy từ đầu tiên của tên lớp, trừ các hãng có tên nhiều từ phải khớp trước:
`Aston Martin`, `Land Rover`, `Rolls-Royce`, `AM General`.

### 3.2. Tách kiểu dáng và số chỗ

Duyệt bảng `BODY_STYLE_RULES` **theo thứ tự**, lấy kết quả khớp đầu tiên:

| Từ khóa | Kiểu dáng | Số chỗ |
| :--- | :--- | ---: |
| `Crew Cab`, `Extended Cab`, `Club Cab`, `Quad Cab`, `SuperCab` | Pickup | 5 |
| `Regular Cab` | Pickup | 3 |
| `Cab` | Pickup | 5 |
| `Minivan` | Minivan | 7 |
| `SUV` | SUV | 7 |
| `Convertible` | Convertible | 4 |
| `Coupe` | Coupe | 4 |
| `Hatchback` | Hatchback | 5 |
| `Wagon` | Wagon | 5 |
| `Sedan` | Sedan | 5 |
| `Van` | Van | 8 |

**Thứ tự là bắt buộc:** `Crew Cab` phải đứng trước `Cab`, nếu không
`Ford F-450 Super Duty Crew Cab 2012` sẽ khớp nhầm. Test
`test_tu_khoa_dai_dung_truoc_tu_khoa_ngan` bảo vệ ràng buộc này.

**14 lớp không chứa từ khóa kiểu dáng** (các bản hiệu năng cao như
`Chevrolet Corvette ZR1 2012`, `Acura TL Type-S 2008`) được ánh xạ tường minh
trong `SPECIAL_BODY_STYLES`.

**Ghi đè số chỗ:** các dòng 2 chỗ (`Corvette`, `Ferrari`, `Lamborghini`,
`smart fortwo`, `Porsche`...) luôn được đặt lại thành 2 chỗ.

### 3.3. Phân khúc

Suy từ hãng xe:

- **exotic** — Ferrari, Lamborghini, Bugatti, McLaren, Rolls-Royce, Bentley,
  Aston Martin, Maybach, Spyker, Fisker, Tesla
- **luxury** — Acura, Audi, BMW, Buick, Cadillac, Infiniti, Jaguar,
  Land Rover, Lincoln, Mercedes-Benz, Volvo
- **economy** — các hãng còn lại

**Ngoại lệ (`SEGMENT_OVERRIDES`):** một số xe thuộc hãng phổ thông nhưng giá
ngang siêu xe. Nếu chỉ dựa vào hãng thì `Chevrolet Corvette ZR1 2012` sẽ bị
xếp economy và định giá 720 triệu — sai hoàn toàn. Các lớp này được ghi đè
tường minh:

| Lớp | Phân khúc |
| :--- | :--- |
| Chevrolet Corvette ZR1 2012 | exotic |
| Chevrolet Corvette Ron Fellows Edition Z06 2007 | exotic |
| Dodge Challenger SRT8 2011 | luxury |
| Dodge Charger SRT-8 2009 | luxury |
| Chrysler 300 SRT-8 2010 | luxury |

## 4. Công thức mô phỏng (simulated)

### 4.0. Biến thiên riêng cho từng dòng xe

Ban đầu công thức chỉ dựa vào (phân khúc, kiểu dáng, năm). Hệ quả: **196
xe chỉ có 87 tổ hợp đặc trưng khác nhau** — 18 xe cùng giá 750 triệu, 15 xe
cùng giá 600 triệu. Module tư vấn vì thế trả về nhiều xe cùng điểm 1.000,
trông như xếp hạng tùy tiện.

Hàm `model_variation` sinh độ lệch riêng cho từng dòng xe:

```python
digest = hashlib.md5(class_name.encode("utf-8")).hexdigest()
unit = int(digest[:6], 16) / 0xFFFFFF * 2 - 1   # đưa về [-1, 1]
return 1 + unit * spread
```

Dùng **hash md5 của tên xe** nên kết quả **luôn tái lập được** — chạy lại
bao nhiêu lần cũng ra đúng con số đó, khác hẳn số ngẫu nhiên.

> Không dùng `hash()` có sẵn của Python: hàm đó thay đổi giữa các lần chạy
> do PYTHONHASHSEED, nên dữ liệu sẽ khác nhau mỗi lần sinh lại.

| Trường | Biên độ lệch |
| :--- | ---: |
| Giá bán | ±18% |
| Mức tiêu hao | ±10% |

Sau khi áp dụng: **196/196 tổ hợp riêng biệt**.

### 4.1. Giá bán (triệu VND)

```text
giá = cơ_sở(phân_khúc) × hệ_số(kiểu_dáng)
      × (1 − 0.02)^(2012 − năm) × biến_thiên
```

| Phân khúc | Giá cơ sở |
| :--- | ---: |
| economy | 600 |
| luxury | 1 800 |
| exotic | 12 000 |

Hệ số kiểu dáng: Hatchback 0.85 · Van 0.95 · Sedan 1.00 · Wagon 1.05 ·
Pickup 1.10 · Minivan 1.15 · Coupe 1.20 · SUV 1.25 · Convertible 1.35

Xe càng cũ càng khấu hao: giảm 2%/năm so với mốc 2012.

### 4.2. Mức tiêu hao (L/100km)

```text
tiêu_hao = cơ_sở(kiểu_dáng) × hệ_số(phân_khúc)
           × (1 + 0.015)^(2012 − năm) × biến_thiên
```

Cơ sở theo kiểu dáng: Hatchback 6.5 · Sedan 7.5 · Wagon 8.0 · Coupe 9.5 ·
Convertible 10.0 · Minivan 10.5 · SUV 11.0 · Van 11.5 · Pickup 12.0

Hệ số phân khúc: economy 1.00 · luxury 1.15 · exotic 1.60

Xe đời cũ tốn nhiên liệu hơn: tăng 1.5%/năm so với mốc 2012.

## 5. Cấu trúc file đầu ra

`data/car_specs.csv` — 196 dòng, mã hóa UTF-8 có BOM (để Excel đọc đúng).

| Cột | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `class_id` | int | 1–196, khớp với nhãn của mô hình phân loại |
| `class_name` | str | Tên lớp gốc Stanford Cars |
| `brand` | str | Hãng xe |
| `model` | str | Dòng xe (đã loại từ khóa kiểu dáng) |
| `body_style` | str | Sedan / SUV / Coupe / ... |
| `year` | int | Năm sản xuất |
| `seats` | int | Số chỗ ngồi |
| `segment` | str | economy / luxury / exotic |
| `price_million_vnd` | float | Giá bán — **mô phỏng** |
| `fuel_l_per_100km` | float | Mức tiêu hao — **mô phỏng** |
| `data_source` | str | `derived+simulated` |

### Phân bố kết quả

- **Phân khúc:** economy 107 · luxury 58 · exotic 31
- **Kiểu dáng:** Sedan 50 · SUV 35 · Coupe 33 · Convertible 26 · Pickup 18 ·
  Hatchback 14 · Wagon 8 · Van 6 · Minivan 6
- **Tổ hợp đặc trưng riêng biệt:** 196/196 (nhờ hàm `model_variation`)

## 6. Cách chạy

```bash
python scripts/generate_car_specs.py
```

Tùy chọn `--output` để đổi đường dẫn file CSV đầu ra.

Chạy kiểm thử:

```bash
python -m unittest discover tests
```

## 7. Nguồn danh sách 196 lớp

`data/stanford_cars_classes.txt` được trích từ danh sách `_NAMES` trong mã
nguồn chính thức của TensorFlow Datasets:

<https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/cars196.py>

Đây là danh sách nguyên bản của dataset, không phải tự gõ lại.

## 8. Hướng thay thế bằng số liệu thật

Nếu sau này cần giá thật, chỉ việc thay nội dung `data/car_specs.csv` mà
**giữ nguyên tên các cột** — toàn bộ module tư vấn không phải sửa dòng nào.
