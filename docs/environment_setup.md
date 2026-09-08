# Cài đặt môi trường

## 1. Phiên bản Python: bắt buộc 3.11

Dự án **phải** dùng Python 3.11 (đang dùng: 3.11.9).

Lý do: các thư viện cốt lõi chưa hỗ trợ Python 3.13+.

| Thư viện | Phiên bản Python cao nhất có wheel |
| :--- | :--- |
| `onnxruntime` | 3.12 |
| `torch` | 3.12 |
| `opencv-python` | 3.12 |

Nếu chạy trên Python 3.13/3.14, `pip install` sẽ thất bại vì không tìm được
wheel — đây là phần cốt lõi của đề tài nên không thể bỏ qua.

> **Cảnh báo về bản Microsoft Store:** lệnh `python` trên Windows có thể trỏ
> tới shortcut của Microsoft Store (một bản Python khác, thường mới hơn).
> Bản này bị giới hạn quyền ghi và hay gây lỗi khó hiểu khi cài package.
>
> Luôn dùng `py -3.11` thay cho `python` khi tạo môi trường ảo. Kiểm tra các
> bản Python đã cài bằng:
>
> ```powershell
> py -0p
> ```

## 2. Tạo môi trường ảo

```powershell
py -3.11 -m venv .venv
```

Thư mục `.venv/` đã được `.gitignore` bỏ qua, không đẩy lên git.

## 3. Cài thư viện

Dự án tách làm hai file requirements:

| File | Dùng ở đâu | Nội dung |
| :--- | :--- | :--- |
| `requirements.txt` | Máy local | Thư viện để **chạy** app (suy luận ONNX, giao diện) |
| `requirements-train.txt` | Colab / Kaggle | Thêm `torch`, `ultralytics`, `skl2onnx` để **huấn luyện** |

Máy local chỉ cần file thứ nhất — không cần cài `torch` (~2.5 GB) vì việc
huấn luyện diễn ra trên Colab:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 4. Kiểm tra môi trường

```powershell
.\.venv\Scripts\python.exe -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

Kết quả mong đợi phải chứa `CPUExecutionProvider`:

```text
['AzureExecutionProvider', 'CPUExecutionProvider']
```

Chạy toàn bộ kiểm thử:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

## 5. Phiên bản đã xác minh

Cấu hình dưới đây đã chạy thành công (17/17 test pass):

| Thành phần | Phiên bản |
| :--- | :--- |
| Python | 3.11.9 |
| onnxruntime | 1.19.2 |
| opencv-python | 4.10.0 |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| scikit-learn | 1.5.2 |
| streamlit | 1.39.0 |

`numpy` được ghim ở nhánh **1.x** (không dùng 2.x) vì `onnxruntime 1.19.2`
và `opencv-python` biên dịch với ABI của numpy 1.x.

## 6. Ghi chú khi chạy lệnh trong PowerShell

PowerShell 5.1 hiển thị mọi thứ trong luồng stderr dưới dạng lỗi đỏ kèm
`NativeCommandError`. Vì `unittest` và module `logging` đều ghi ra stderr,
việc thấy chữ đỏ khi chạy test là **bình thường** — cứ nhìn dòng `OK` hoặc
`FAILED` ở cuối để biết kết quả thật.

## 7. Môi trường huấn luyện (Colab / Kaggle)

Máy local không có GPU rời nên toàn bộ việc huấn luyện chạy trên Colab.
Notebook sẽ tự cài `requirements-train.txt`. Chi tiết ghi trong tài liệu của
bước huấn luyện.
