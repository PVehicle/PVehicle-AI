# So sánh Pandas và ONNX cho module tư vấn

## 1. Câu hỏi cần trả lời

Đề cương ban đầu yêu cầu:

> Chuyển đổi pipeline của Scikit-learn sang định dạng ONNX bằng thư viện
> `skl2onnx` để đồng bộ hoàn toàn kiến trúc chạy local.

Câu hỏi đặt ra: **đồng bộ toàn bộ kiến trúc sang ONNX có thực sự đem lại
lợi ích không?** Tài liệu này trả lời bằng số liệu đo được, không phải suy
đoán.

## 2. Cách thực hiện

Triển khai song song hai bản của cùng một chức năng "tìm xe tương tự":

| Bản | Cách làm |
| :--- | :--- |
| **Pandas + NumPy** | Tính khoảng cách Euclid trực tiếp bằng NumPy |
| **ONNX Runtime** | Đóng gói `KNeighborsTransformer` bằng `skl2onnx` |

Cả hai dùng **chung một ma trận đặc trưng** (196 xe × 16 chiều) để kết quả
so sánh được với nhau.

### Kiểm tra tính đúng đắn trước

Trước khi đo tốc độ, script xác minh hai bản cho **kết quả giống hệt nhau**
trên 3 xe mẫu. Nếu kết quả khác nhau thì số liệu tốc độ vô nghĩa — nhanh hơn
mà sai thì không dùng được.

Kết quả: **hai bản trả về danh sách xe giống hệt nhau.**

## 3. Kết quả đo

Máy đo: CPU, Python 3.11.9, ONNX Runtime 1.19.2, 200 lần chạy sau 20 lần
làm nóng.

| Bản triển khai | Trung vị | Độ lệch chuẩn |
| :--- | ---: | ---: |
| Pandas + NumPy | **0.698 ms** | 0.199 ms |
| ONNX Runtime (KNN) | 0.823 ms | 0.217 ms |

- **Bản Pandas nhanh hơn 1.18 lần**
- Chi phí khởi tạo `InferenceSession`: **2.8 ms** (chỉ một lần)
- Kích thước file ONNX: 13.5 KB

> Dùng **trung vị** thay vì trung bình: phép đo thời gian trên máy tính cá
> nhân hay bị nhiễu bởi các tiến trình khác, trung vị ít bị ảnh hưởng hơn.

## 4. Kết luận

**Với module tư vấn, ONNX không đem lại lợi ích.** Bản Pandas đơn giản hơn
và nhanh hơn.

### Vì sao?

ONNX Runtime có chi phí cố định cho mỗi lần gọi: chuyển dữ liệu vào/ra
runtime, kiểm tra kiểu, cấp phát bộ nhớ. Với bài toán chỉ có **196 dòng dữ
liệu và 16 đặc trưng**, phần tính toán thực sự quá nhỏ — chi phí cố định
lấn át hoàn toàn.

Đây **không phải là thất bại của ONNX**, mà là một kết luận đúng về phạm vi
áp dụng của nó.

### ONNX có lợi ở đâu?

Ngược lại hoàn toàn với tầng nhận diện của dự án này:

| | Module tư vấn | Mô hình phân loại ảnh |
| :--- | :--- | :--- |
| Kích thước | 196 × 16 | ~5,3 triệu tham số |
| Phép tính | Vài nghìn phép | Hàng tỷ phép |
| Chi phí cố định | Chiếm phần lớn | Không đáng kể |
| Lợi ích của ONNX | **Không có** | **Rất lớn** |

Với mô hình phân loại, ONNX cho phép chạy mà **không cần cài PyTorch**
(~2,5 GB) trên máy local — đây mới là giá trị thật.

### Giới hạn của việc đóng gói

Chỉ **bước tìm láng giềng gần nhất** đóng gói được sang ONNX. Việc lọc theo
ràng buộc cứng (ngân sách, số chỗ, kiểu dáng) vẫn phải làm bằng Pandas, vì
ONNX không biểu diễn được phép lọc động như vậy.

Nghĩa là dù chọn ONNX thì cũng **không loại bỏ được Pandas** khỏi hệ thống —
mục tiêu "đồng bộ hoàn toàn kiến trúc" không đạt được trọn vẹn.

## 5. Quyết định cho dự án

Ứng dụng dùng **bản Pandas** (`src/rec/recommender.py`) làm mặc định.

Bản ONNX (`src/rec/onnx_recommender.py`) vẫn được giữ lại vì:

- Chứng minh đã thực hiện đúng yêu cầu của đề cương
- Cung cấp số liệu thực nghiệm cho chương đánh giá
- Là ví dụ minh họa cách dùng `skl2onnx`

## 6. Cách chạy lại phép đo

```powershell
.\.venv\Scripts\python.exe scripts/export_recommender_onnx.py
```

Script sẽ export mô hình, kiểm tra tính đúng đắn, rồi in bảng so sánh.

Thêm `--skip-benchmark` nếu chỉ muốn export.

## 7. Một lỗi gặp phải khi export

Ban đầu dùng `NearestNeighbors`, `skl2onnx` báo lỗi:

```text
RuntimeError: The model defines radius and n_neighbors at the same time
(1.0 and 6). This case is not supported.
```

Nguyên nhân: `NearestNeighbors` đặt giá trị mặc định cho **cả** `radius`
lẫn `n_neighbors`, mà `skl2onnx` không xử lý được trường hợp này.

Cách sửa: dùng `KNeighborsTransformer` — lớp chỉ có `n_neighbors`.

Lưu ý thêm về định dạng đầu ra: `KNeighborsTransformer` trả về **một ma
trận khoảng cách dày `[1, 196]`** (chỉ các láng giềng có giá trị khác 0),
chứ không phải cặp `(khoảng cách, chỉ số)` như `NearestNeighbors.kneighbors`
trong Scikit-learn.
