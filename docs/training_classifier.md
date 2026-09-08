# Huấn luyện tầng phân loại xe (196 lớp)

## 1. Vị trí trong kiến trúc

Hệ thống nhận diện gồm **2 tầng**:

| Tầng | Mô hình | Huấn luyện |
| :--- | :--- | :--- |
| 1. Phát hiện xe | YOLOv8 pretrained COCO | **Không cần train** |
| 2. Phân loại dòng xe | EfficientNet-B0 (196 lớp) | Notebook này |

Lý do tách 2 tầng: Stanford Cars **không có bounding box theo từng dòng xe**
— mỗi ảnh chỉ có một hộp bao duy nhất với nhãn là dòng xe. Không thể train
YOLOv8 ra thẳng 196 lớp từ dữ liệu này. Ngược lại, dùng YOLOv8 pretrained
COCO (đã biết class `car`/`truck`/`bus`) để khoanh vùng, rồi train riêng một
mô hình phân loại trên ảnh đã crop, vừa đúng dữ liệu vừa tiết kiệm rất nhiều
thời gian.

## 2. Notebook

`notebooks/train_classifier_colab.ipynb` — chạy trên **Google Colab**
(Runtime → Change runtime type → **T4 GPU**).

> Notebook được sinh ra từ `scripts/build_train_notebook.py`, không sửa
> trực tiếp file `.ipynb`. Lý do: JSON của notebook rất khó đọc trong git
> diff. Muốn đổi nội dung thì sửa script rồi chạy lại:
>
> ```powershell
> .\.venv\Scripts\python.exe scripts/build_train_notebook.py
> ```

## 3. Nguồn dữ liệu

Trang gốc của Stanford (`ai.stanford.edu/~jkrause/cars/`) **đã bị gỡ bỏ**,
khiến cả `torchvision.datasets.StanfordCars` cũng không tải được nữa
([pytorch/vision#7545](https://github.com/pytorch/vision/issues/7545)).

Dự án dùng bản mirror trên Kaggle:

```text
rickyyyyyyy/torchvision-stanford-cars   (~2 GB)
```

### Vì sao chọn bản này

| Yêu cầu | Bản `rickyyyyyyy` | Bản `jutrera` |
| :--- | :--- | :--- |
| Bounding box | **Có** (giữ file `.mat` gốc) | Không |
| Tên lớp | `Tesla Model S 2012` | `2012 Tesla Model S` |
| Khớp `car_specs.csv` | **100%** | Phải viết hàm ánh xạ |

Bounding box là **bắt buộc** — không có nó thì không crop được cho tầng
phân loại. Bản `jutrera` chỉ chia ảnh vào thư mục theo lớp và đã mất toàn bộ
tọa độ hộp bao.

### Cấu trúc sau khi giải nén

```text
data/stanford_cars/
├── cars_train/                        # 8.144 ảnh
├── cars_test/                         # 8.041 ảnh
├── cars_test_annos_withlabels.mat     # nhãn + bbox tập test
└── devkit/
    ├── cars_train_annos.mat           # nhãn + bbox tập train
    └── cars_meta.mat                  # tên 196 lớp
```

## 4. Chuẩn bị tài khoản Kaggle

1. Tạo tài khoản tại <https://www.kaggle.com>
2. Vào **Settings** (ảnh đại diện) → mục **API**
3. Bấm **Create New Token**
4. Copy chuỗi token (dạng `KGAT_...`), dán vào ô nhập trong notebook

### Hai cơ chế xác thực

Kaggle đã chuyển từ file `kaggle.json` sang **token dạng chuỗi**. Notebook
hỗ trợ cả hai, tìm theo đúng thứ tự mà thư viện `kaggle` tự tìm:

| Thứ tự | Nguồn | Ghi chú |
| ---: | :--- | :--- |
| 1 | Biến môi trường `KAGGLE_API_TOKEN` | Cơ chế mới |
| 2 | `~/.kaggle/access_token` | Cơ chế mới, dạng file |
| 3 | `~/.kaggle/kaggle.json` | Cơ chế cũ, vẫn dùng được |

Nếu đã có sẵn một trong ba, notebook dùng luôn và không hỏi gì.

### Bảo mật token

Ô nhập token dùng `getpass`, nên token **không hiện ra màn hình và không bị
lưu vào file notebook**.

> **Tuyệt đối không gõ thẳng token vào ô code.** Notebook lưu cả kết quả
> chạy, nên token sẽ lộ ra khi chia sẻ file `.ipynb` hoặc commit lên git.

Kaggle chỉ cho xem token đúng một lần. Nếu lỡ để lộ (chụp màn hình, dán
nhầm vào chat, commit lên git), vào lại **Settings → API → Create New
Token** để sinh token mới — token cũ tự hết hiệu lực ngay.

## 5. Tiền xử lý: crop theo bounding box

Tầng phân loại chỉ cần vùng chứa xe. Ảnh được crop **một lần** rồi lưu ra
đĩa, thay vì crop lại ở mỗi epoch.

Hộp bao được **nới rộng 8%** mỗi chiều:

- Giữ lại một phần bối cảnh giúp mô hình nhận dạng tốt hơn
- Tránh cắt mất viền xe khi hộp bao hơi lệch

Tọa độ luôn được ép nằm trong khung ảnh để không lỗi khi hộp chạm viền.

## 6. Cấu hình huấn luyện

| Tham số | Giá trị | Ghi chú |
| :--- | :--- | :--- |
| Kiến trúc | EfficientNet-B0 | ~5,3 triệu tham số, nhẹ khi chạy CPU |
| Trọng số khởi tạo | ImageNet | Transfer learning |
| Kích thước ảnh | 224×224 | |
| Batch size | 64 | |
| Epoch | 20 | ~45-60 phút trên T4 |
| Optimizer | AdamW (lr 1e-3, wd 1e-4) | |
| Scheduler | CosineAnnealingLR | |
| Label smoothing | 0.1 | Chống overfit trên lớp giống nhau |
| Mixed precision | Có (`torch.amp`) | Nhanh ~2× trên T4 |

### Tăng cường dữ liệu (chỉ tập train)

| Phép biến đổi | Mục đích |
| :--- | :--- |
| `RandomResizedCrop` | Chịu được thay đổi khoảng cách chụp |
| `RandomHorizontalFlip` | Xe chụp từ trái/phải đều nhận được |
| `ColorJitter` | Chịu được điều kiện ánh sáng khác nhau |
| `RandomErasing` | Chịu được trường hợp xe bị che một phần |

Tập test **không** tăng cường — chỉ resize và chuẩn hóa, để kết quả đánh giá
phản ánh đúng năng lực thật.

## 7. Checkpoint và resume — phần quan trọng nhất

Colab free ngắt phiên sau khoảng **4-6 giờ** và xóa sạch ổ đĩa tạm. Nếu
không lưu checkpoint thì mỗi lần ngắt là mất trắng.

Notebook lưu trạng thái đầy đủ ra **Google Drive** sau **mỗi epoch**:

```text
MyDrive/PVehicle-AI/checkpoints/
├── last.pt      # epoch gần nhất — dùng để học tiếp
└── best.pt      # top-1 cao nhất — dùng để export ONNX
```

Mỗi checkpoint chứa: `model_state`, `optimizer_state`, `scheduler_state`,
số epoch, lịch sử huấn luyện, độ chính xác tốt nhất và danh sách nhãn.

**Khi Colab ngắt:** chạy lại notebook từ đầu. Ô huấn luyện tự đọc `last.pt`
và học tiếp từ đúng chỗ đang dở, không mất tiến độ.

## 8. Đánh giá

Báo cáo cả **top-1** và **top-5**.

Với bài toán 196 lớp chi tiết (nhiều dòng xe khác nhau rất ít, ví dụ
`Audi S4 Sedan 2007` và `Audi S4 Sedan 2012`), top-5 phản ánh độ hữu ích
thực tế tốt hơn — người dùng nhìn 5 gợi ý là nhận ra xe của mình.

### Kỳ vọng kết quả

Với EfficientNet-B0 + 20 epoch trên Colab free, mức hợp lý là **top-1
khoảng 80-88%**, top-5 thường trên 95%.

Các baseline công bố đạt ~85-93% top-1 nhưng dùng backbone lớn hơn và train
lâu hơn nhiều. Đây là đồ án học tập nên không đặt mục tiêu vượt baseline.

## 9. Export ONNX

Nạp lại `best.pt` (**không** phải epoch cuối) rồi export.

### Quyết định về `dynamic_axes`

Chỉ để **batch** động, còn chiều rộng/cao **cố định 224×224**:

```python
dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}}
```

Tài liệu đề cương ban đầu ghi `dynamic=True` cho toàn bộ trục. Nhưng ảnh đầu
vào luôn được resize về 224 trước khi suy luận, nên trục động ở chiều
rộng/cao chỉ làm ONNX Runtime khó tối ưu hơn mà không được lợi ích gì. Riêng
batch để động là cần thiết — một ảnh có thể chứa nhiều xe.

### Kiểm tra sau khi export

Notebook đối chiếu kết quả PyTorch và ONNX Runtime trên cùng đầu vào, yêu
cầu sai lệch tuyệt đối lớn nhất `< 1e-3`. Nếu vượt ngưỡng thì file ONNX bị
lỗi — phát hiện ngay tại đây thay vì lúc chạy app.

## 10. Kết quả cần tải về

| File | Đặt vào | Nội dung |
| :--- | :--- | :--- |
| `car_classifier.onnx` | `models/car_classifier.onnx` | Mô hình phân loại |
| `class_names.json` | `models/class_names.json` | 196 nhãn theo đúng thứ tự |

`class_names.json` là **bắt buộc**: nó cho biết chỉ số đầu ra của mô hình
ứng với tên xe nào, và chính là khóa để nối sang cột `class_name` trong
`data/car_specs.csv` của module tư vấn.

> Cả hai file đều nằm sẵn trong `MyDrive/PVehicle-AI/`, tải trực tiếp từ đó
> nếu trình duyệt chặn lệnh `files.download()`.

## 11. Một lỗi dễ mắc: thứ tự nhãn

`ImageFolder` sắp xếp thư mục theo **thứ tự chuỗi**, không phải thứ tự số:

```text
'0', '1', '10', '100', '101', ..., '2', '20', ...
```

Nghĩa là chỉ số nội bộ của `ImageFolder` **không trùng** với `class_id` gốc.
Notebook xử lý bằng cách dựng lại ánh xạ `FOLDER_TO_CLASS` và sinh danh sách
`IDX_TO_NAME` theo đúng thứ tự mà mô hình thực sự học.

Nếu bỏ qua bước này, mô hình vẫn train bình thường và độ chính xác vẫn cao,
nhưng **mọi tên xe hiển thị ra đều sai** — một lỗi rất khó phát hiện.

## 12. Nguồn tham khảo

- Krause et al., *3D Object Representations for Fine-Grained Categorization*,
  3dRR-13, ICCV 2013 — <https://ai.stanford.edu/~jkrause/papers/3drr13.pdf>
- Danh sách 196 lớp: xem `docs/car_specs_generation.md`, mục 7
