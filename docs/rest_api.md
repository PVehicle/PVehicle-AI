# REST API

API cho phép ứng dụng khác (mobile, web, hệ thống bên thứ ba) sử dụng khả
năng nhận diện và tư vấn xe của dự án.

## 1. Khởi chạy

### Chế độ phát triển

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

Mở <http://localhost:8000/docs> để xem tài liệu tương tác (Swagger UI) —
gọi thử được ngay trên trình duyệt.

### Chế độ chạy thật

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
cp .env.example .env      # rồi sửa các giá trị bên trong
docker compose up --build
```

## 2. Cấu hình

Mọi cấu hình đọc từ biến môi trường hoặc file `.env`, **không viết cứng
trong mã nguồn**. Chép `.env.example` thành `.env` rồi sửa.

| Biến | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `PVEHICLE_API_KEYS` | *(trống)* | Danh sách API key, ngăn bởi dấu phẩy |
| `PVEHICLE_CORS_ORIGINS` | `*` | Origin được phép gọi từ trình duyệt |
| `PVEHICLE_MAX_UPLOAD_BYTES` | 10 MB | Kích thước file tối đa |
| `PVEHICLE_MAX_IMAGE_PIXELS` | 50 triệu | Số điểm ảnh tối đa |
| `PVEHICLE_RATE_LIMIT_DEFAULT` | `60/minute` | Hạn mức chung |
| `PVEHICLE_RATE_LIMIT_RECOGNIZE` | `20/minute` | Hạn mức cho nhận diện |
| `PVEHICLE_INFERENCE_WORKERS` | 2 | Số pipeline chạy song song |

> ⚠️ **`PVEHICLE_API_KEYS` để trống nghĩa là API mở công khai.** Khi khởi
> động, log sẽ cảnh báo. Chỉ chấp nhận điều này khi phát triển trên máy
> mình.

Sinh API key ngẫu nhiên:

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 3. Xác thực

Khi máy chủ có cấu hình API key, mọi request phải kèm header:

```http
X-API-Key: <key của bạn>
```

Hai endpoint `/health` và `/ready` **luôn mở** — bộ cân bằng tải phải gọi
được mà không cần key.

## 4. Danh sách endpoint

### Kiểm tra sức khỏe

| Method | Đường dẫn | Mô tả |
| :--- | :--- | :--- |
| GET | `/health` | Tiến trình còn sống không (liveness) |
| GET | `/ready` | Đã nạp đủ mô hình chưa (readiness) |

`/ready` trả **503** khi chưa nạp được mô hình ONNX. Lúc đó các endpoint
tra cứu và tư vấn **vẫn hoạt động** vì chúng chỉ cần `car_specs.csv`.

### Nhận diện

| Method | Đường dẫn | Mô tả |
| :--- | :--- | :--- |
| POST | `/api/v1/recognize` | Nhận diện xe từ ảnh tải lên |

Tham số truy vấn:

| Tham số | Mặc định | Ý nghĩa |
| :--- | ---: | :--- |
| `top_k` | 5 | Số dự đoán trả về mỗi xe (1-20) |
| `include_similar` | false | Có kèm gợi ý xe tương tự không |
| `similar_count` | 3 | Số xe tương tự (1-10) |

```bash
curl -X POST "http://localhost:8000/api/v1/recognize?top_k=3" \
  -H "X-API-Key: your-key" \
  -F "file=@anh_xe.jpg"
```

### Danh mục xe

| Method | Đường dẫn | Mô tả |
| :--- | :--- | :--- |
| GET | `/api/v1/cars` | Liệt kê, lọc, phân trang |
| GET | `/api/v1/cars/filters` | Các giá trị hợp lệ để lọc |
| GET | `/api/v1/cars/{class_name}` | Thông số một dòng xe |
| GET | `/api/v1/cars/{class_name}/similar` | Các xe tương tự |
| POST | `/api/v1/recommend` | Gợi ý theo nhu cầu |

```bash
# Gợi ý xe 7 chỗ dưới 800 triệu
curl -X POST "http://localhost:8000/api/v1/recommend" \
  -H "Content-Type: application/json" \
  -d '{"max_price": 800, "min_seats": 7, "top_n": 5}'
```

## 5. Hai hệ đánh số — điểm dễ nhầm

Response của `/recognize` chứa **hai trường số khác nhau**:

| Trường | Nằm trong | Ý nghĩa | Ví dụ |
| :--- | :--- | :--- | ---: |
| `model_index` | `predictions` | Chỉ số đầu ra của mô hình (0-195) | 95 |
| `class_id` | `specs` | Số thứ tự gốc Stanford Cars (1-196) | 185 |

Cùng chỉ một chiếc `Tesla Model S Sedan 2012` nhưng **hai con số khác
nhau**. Nguyên nhân: `ImageFolder` sắp xếp thư mục theo thứ tự chuỗi
(`'0'`, `'1'`, `'10'`, `'100'`...) chứ không theo thứ tự số, nên chỉ số
nội bộ của mô hình lệch khỏi số thứ tự gốc.

> **Luôn dùng `class_name` để đối chiếu giữa hai nơi**, đừng dùng số.

## 6. Mã lỗi

| Mã | Khi nào | Cách xử lý |
| ---: | :--- | :--- |
| 400 | File rỗng hoặc ảnh hỏng | Kiểm tra lại file |
| 401 | Thiếu hoặc sai API key | Kiểm tra header `X-API-Key` |
| 404 | Không tìm thấy dòng xe | Kiểm tra `class_name` |
| 413 | File hoặc ảnh quá lớn | Giảm kích thước ảnh |
| 415 | Định dạng không hỗ trợ | Dùng JPEG, PNG, BMP, GIF, WEBP |
| 422 | Dữ liệu gửi lên không hợp lệ | Đọc trường `detail` |
| 429 | Vượt hạn mức gọi | Chờ rồi thử lại |
| 503 | Mô hình chưa sẵn sàng | Kiểm tra thư mục `models/` |

Mọi lỗi trả về cùng khuôn dạng:

```json
{
  "detail": "Mô tả lỗi bằng tiếng Việt",
  "request_id": "a1b2c3d4e5f6"
}
```

`request_id` cũng nằm trong header `X-Request-ID` của mọi response — dùng
nó để tra log khi cần hỗ trợ.

Khi gặp **429**, response kèm header `Retry-After` (giây) cho biết nên chờ
bao lâu.

> Cả hai header đều được khai báo trong `Access-Control-Expose-Headers`,
> nên JavaScript ở frontend đọc được. Thiếu khai báo này thì trình duyệt
> vẫn nhận header nhưng **giấu khỏi JavaScript**.

## 7. Các quyết định kỹ thuật

### ONNX Runtime không an toàn đa luồng

Đây là ràng buộc quan trọng nhất khi đưa mô hình lên API.

Gọi song song trên cùng một `InferenceSession` có thể cho **kết quả sai
hoặc làm sập tiến trình**. Cách xử lý trong dự án:

- Tạo sẵn một **nhóm pipeline**, mỗi cái giữ session riêng
- Mỗi request **mượn một pipeline** từ hàng đợi, trả lại sau khi xong
- Request đến khi hết pipeline sẽ **chờ đến lượt** thay vì dùng chung

Số pipeline đặt bằng `PVEHICLE_INFERENCE_WORKERS`. Mỗi cái tốn khoảng
30 MB RAM.

### Suy luận chạy trong luồng riêng

Suy luận là tác vụ nặng CPU, chạy thẳng trong hàm `async` sẽ **chặn vòng
lặp sự kiện** khiến mọi request khác phải đợi. Code dùng
`asyncio.to_thread` để đẩy phần tính toán sang luồng riêng.

### Không tin `content-type` do client khai báo

Client có thể đặt `content-type: image/jpeg` cho một file bất kỳ. API kiểm
tra **chữ ký byte đầu file** để xác định định dạng thật.

### Chặn file lớn ngay khi đang đọc

`read_upload` đọc theo từng khối 1 MB và dừng ngay khi vượt hạn mức, thay
vì đọc hết rồi mới kiểm tra — tránh việc một file rất lớn làm cạn bộ nhớ
máy chủ.

### So sánh API key chống timing attack

Dùng `secrets.compare_digest` thay vì `==`. Phép so sánh chuỗi thông
thường thoát sớm ở ký tự đầu tiên khác nhau, để lộ thông tin qua thời gian
phản hồi.

### Mô hình nạp một lần lúc khởi động

Nạp trong `lifespan` chứ không phải mỗi request. Mô hình nặng ~30 MB và
mất vài giây để khởi tạo session.

Nếu thiếu file mô hình, API **vẫn khởi động được**: các endpoint tra cứu
và tư vấn hoạt động bình thường, riêng `/recognize` trả 503.

## 8. Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_api -v
```

22 test, bao gồm: health check, phân trang, lọc, ràng buộc cứng của bộ lọc
giá, từ chối file giả mạo, nhận diện chữ ký ảnh.

Các test cần mô hình ONNX tự bỏ qua nếu `models/` còn trống.

## 9. Giới hạn hiện tại

**Rate limit lưu trong bộ nhớ tiến trình.** Chạy nhiều bản sao thì mỗi bản
đếm riêng. Muốn dùng chung phải chuyển sang Redis.

**Chưa có xác thực người dùng.** Chỉ có API key ở mức ứng dụng, không phân
biệt từng người dùng cuối.

**Chưa lưu lịch sử.** Mỗi request độc lập, không ghi lại kết quả.

**Một worker uvicorn.** Docker chạy `--workers 1` vì mỗi worker sẽ nạp
riêng bộ mô hình, nhân đôi bộ nhớ. Muốn mở rộng nên chạy nhiều container
sau một bộ cân bằng tải.
