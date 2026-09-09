# Kết quả đánh giá hệ thống

Số liệu trong tài liệu này lấy từ mô hình đã huấn luyện thật, đo ngày
09/09/2026.

## 1. Kết quả huấn luyện tầng phân loại

Mô hình: EfficientNet-B0, 196 lớp, huấn luyện 19 epoch trên Google Colab
(GPU T4).

| Chỉ số | Kết quả |
| :--- | ---: |
| **Top-1 accuracy** | **83.02%** |
| **Top-5 accuracy** | **95.26%** |
| Epoch tốt nhất | 19 |
| Train accuracy (epoch cuối) | 99.41% |
| Chênh lệch train − val | 16.38% |

![Biểu đồ huấn luyện](../reports/training_curves.png)

### Nhận xét

**Kết quả đạt kỳ vọng.** Mức 83% top-1 nằm trong khoảng dự kiến 80-88% cho
EfficientNet-B0 với 20 epoch. Các baseline công bố đạt 85-93% nhưng dùng
backbone lớn hơn và huấn luyện lâu hơn nhiều.

**Top-5 95.26% phản ánh độ hữu ích thực tế.** Với 196 lớp rất giống nhau
(ví dụ `Audi S4 Sedan 2007` và `Audi S4 Sedan 2012`), việc đưa xe đúng vào
5 gợi ý đầu đã đủ dùng cho người dùng cuối. Cứ 20 lần thì 19 lần hệ thống
đưa ra đáp án đúng trong danh sách.

**Có hiện tượng quá khớp nhẹ.** Chênh lệch train − val là 16.38%, vượt
ngưỡng cảnh báo 15%:

- Train accuracy đạt 99.41% — mô hình gần như thuộc lòng tập huấn luyện
- Loss validation phẳng dần từ epoch 15, không giảm thêm

Đây là điều **bình thường** với bài toán fine-grained trên tập dữ liệu chỉ
8.144 ảnh huấn luyện chia cho 196 lớp (trung bình ~42 ảnh/lớp).

Nếu muốn cải thiện:

- Tăng cường dữ liệu mạnh hơn (MixUp, CutMix, RandAugment)
- Dừng sớm ở epoch 14-15 khi val loss bắt đầu phẳng
- Dùng backbone lớn hơn (EfficientNet-B2) với dropout cao hơn

## 2. Kiểm chứng mô hình sau khi export ONNX

| Kiểm tra | Kết quả |
| :--- | :--- |
| Kích thước file | 17.02 MB (một file duy nhất) |
| Số node trong đồ thị | 239 |
| Tổng tham số | 4.237.616 |
| Đầu vào | `input` `[batch, 3, 224, 224]` |
| Đầu ra | `logits` `[batch, 196]` |
| Batch động | Hoạt động (thử batch 1 và 4) |
| NaN / Inf trong đầu ra | Không có |
| Tổng xác suất sau softmax | 1.000000 |

### Kiểm tra thứ tự nhãn

`class_names.json` khớp **tập hợp** với danh sách 196 lớp gốc, nhưng khác
**thứ tự ở 194/196 vị trí**.

Đây là điều đã lường trước: `ImageFolder` sắp xếp thư mục theo thứ tự chuỗi
(`'0'`, `'1'`, `'10'`, `'100'`, ...) chứ không theo thứ tự số. Notebook đã
lưu lại đúng thứ tự mà mô hình thực sự học.

> **Nếu bỏ qua bước này**, mô hình vẫn train bình thường và độ chính xác
> vẫn 83%, nhưng **194/196 tên xe hiển thị ra sẽ sai**. Đây là loại lỗi rất
> khó phát hiện vì mọi thứ trông vẫn hoạt động.

Đối chiếu với `car_specs.csv`: **0 nhãn thiếu** — mọi dòng xe nhận diện
được đều tra cứu được thông số.

## 3. Kiểm tra hành vi với ảnh không phải ô tô

Thử với 3 loại ảnh vô nghĩa:

| Ảnh đầu vào | Phát hiện xe | Xác suất cao nhất | Đủ tin cậy |
| :--- | :--- | ---: | :--- |
| Nhiễu ngẫu nhiên | Không | 2.17% | Không |
| Xám trơn | Không | 1.32% | Không |
| Hình chữ nhật đậm | Không | 7.58% | Không |

Trên 20 ảnh nhiễu ngẫu nhiên: xác suất cao nhất trung bình **2.26%**
(ngưỡng đoán mò với 196 lớp là 0.51%).

**Kết luận:** mô hình "biết mình không biết". Cơ chế phát hiện ảnh không
phải ô tô (`MIN_CONFIDENCE = 0.15`) kích hoạt đúng trong cả ba trường hợp.

## 4. Kiểm tra với ảnh xe thật

Ba ảnh lấy từ Wikimedia Commons, hoàn toàn nằm ngoài tập huấn luyện:

| Ảnh | Kết quả nhận diện | Độ tin cậy | Đúng? |
| :--- | :--- | ---: | :---: |
| Jeep Wrangler 2012 | Jeep Wrangler SUV 2012 | 45.14% | ✅ |
| Honda Odyssey 2011 | Honda Odyssey Minivan 2012 | 85.49% | ✅ |
| Tesla Model S | Tesla Model S Sedan 2012 | 53.49% | ✅ |

### Nhận xét

**Cả ba đều đúng ngay top-1.** Đáng chú ý là ảnh Honda Odyssey 2011 được
nhận đúng thành lớp `Honda Odyssey Minivan 2012` — đời xe lệch một năm
nhưng cùng thế hệ, đây là kết quả đúng về mặt thực tiễn.

**Xử lý được nhiều xe trong một ảnh.** Ảnh Tesla có 4 xe trong khung, hệ
thống tách và phân loại riêng từng xe.

**Ngưỡng tin cậy hoạt động đúng với xe ở xa.** Một xe nền trong ảnh Tesla
(hộp bao chỉ 121×154 px) chỉ đạt 13.05% và bị đánh dấu "dưới ngưỡng đáng
tin" — đúng như thiết kế, xe mờ ở xa thì không nên khẳng định.

## 5. Hiệu năng suy luận trên CPU

Môi trường đo: Intel Core i7-10510U, Python 3.11.9, ONNX Runtime 1.19.2,
`CPUExecutionProvider`. Ảnh đầu vào 960×661, 30 lần chạy.

| Bước xử lý | Trung vị | Trung bình | Độ lệch | p95 |
| :--- | ---: | ---: | ---: | ---: |
| Tầng 1 — YOLOv8n (640px) | 83.45 ms | 79.86 ms | 13.48 ms | 96.87 ms |
| Tầng 2 — EfficientNet-B0 (224px) | 20.26 ms | 20.03 ms | 1.88 ms | 21.50 ms |
| **Cả luồng (end-to-end)** | **134.84 ms** | 136.57 ms | 13.41 ms | 149.00 ms |

### Phân tích

| Chỉ số | Giá trị |
| :--- | ---: |
| Thông lượng | **7.4 ảnh/giây** |
| Tỷ trọng tầng 1 | 62% |
| Tỷ trọng tầng 2 | 15% |
| Chi phí còn lại (crop, đổi màu, ghép kết quả) | 23% |

**Tầng phát hiện chiếm phần lớn thời gian** (62%) dù mô hình nhỏ hơn tầng
phân loại. Nguyên nhân: đầu vào 640×640 lớn gấp 8 lần diện tích so với
224×224 của tầng phân loại.

**135 ms/ảnh là đủ nhanh** cho kịch bản người dùng tải ảnh lên qua giao
diện web. Nếu cần xử lý video thời gian thực thì phải tối ưu thêm (giảm
kích thước đầu vào, hoặc dùng OpenVINO trên CPU Intel).

## 6. Cách tái lập các số liệu này

```powershell
# Báo cáo huấn luyện (bảng số liệu + biểu đồ)
.\.venv\Scripts\python.exe scripts/evaluate_model.py --checkpoint models/best.pt

# Hiệu năng suy luận
.\.venv\Scripts\python.exe scripts/benchmark_inference.py --runs 30 --image anh.jpg

# Nhận diện một thư mục ảnh
.\.venv\Scripts\python.exe scripts/recognize.py thu_muc_anh/ --top-k 3
```

Kết quả được ghi vào thư mục `reports/` (không đẩy lên git).

## 7. Tổng kết

| Hạng mục | Trạng thái |
| :--- | :--- |
| Độ chính xác top-1 | 83.02% — đạt kỳ vọng |
| Độ chính xác top-5 | 95.26% — tốt |
| Kiểm chứng ONNX | Đầu ra hợp lệ, batch động hoạt động |
| Thứ tự nhãn | Đúng, đối chiếu 0 lỗi với bảng thông số |
| Xử lý ảnh không phải xe | Hoạt động đúng |
| Nhận diện ảnh thật | 3/3 đúng ngay top-1 |
| Hiệu năng CPU | 135 ms/ảnh, 7.4 ảnh/giây |
| Kiểm thử tự động | 69/69 test pass |
