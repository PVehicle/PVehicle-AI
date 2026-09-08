# PVehicle-AI 🚗🤖

**PVehicle-AI** là một hệ thống Trí tuệ Nhân tạo đa nhiệm, tích hợp khả năng nhận diện chính xác các dòng xe ô tô qua hình ảnh và hệ thống tư vấn (Recommendation System) đề xuất dòng xe phù hợp dựa trên nhu cầu cá nhân hóa. 

Dự án được thiết kế đặc thù để tối ưu hóa hiệu suất khi vận hành tại môi trường cục bộ (Local Environment) mà không cần cấu hình phần cứng quá khủng, thông qua việc chuyển đổi và chạy suy luận bằng định dạng **ONNX**.

---

## 🌟 Tính năng chính (Key Features)

- **🔍 Nhận diện phương tiện (Object Detection & Classification):** 
  Sử dụng YOLOv8 để khoanh vùng và phân loại chính xác hãng xe, dòng xe từ ảnh chụp thực tế.
- **💡 Tư vấn thông minh (Recommendation Engine):** 
  Đề xuất xe dựa trên các tiêu chí đầu vào của người dùng (ngân sách, số chỗ ngồi, mục đích sử dụng).
- **⚡ Suy luận tốc độ cao (Fast Inference):** 
  Tối ưu hóa tốc độ xử lý trên CPU/GPU local bằng ONNX Runtime, hoàn toàn độc lập với các framework huấn luyện gốc (PyTorch/TensorFlow).
- **🖥️ Giao diện trực quan (Web-based UI):** 
  Tương tác dễ dàng qua giao diện được xây dựng bằng Streamlit.

---

## 🛠️ Công nghệ sử dụng (Tech Stack)

- **Ngôn ngữ lõi:** Python 3.9+
- **Thị giác máy tính (CV):** YOLOv8 (Ultralytics), OpenCV, PyTorch
- **Machine Learning:** Scikit-learn, Pandas
- **Tối ưu & Triển khai:** ONNX, ONNX Runtime
- **Web App UI:** Streamlit

---

## 📂 Cấu trúc thư mục (Project Structure)

```text
PVehicle-AI/
│
├── data/                  # Thư mục chứa dataset (ảnh test, file csv thông số xe)
├── models/                # Nơi lưu trữ các file mô hình (.pt, .onnx)
│   ├── yolov8n.onnx       # Mô hình nhận diện đã được export
│   └── recommend.onnx     # Mô hình tư vấn (tùy chọn)
│
├── src/                   # Thư mục chứa mã nguồn chính
│   ├── cv_module.py       # Xử lý hình ảnh và chạy suy luận YOLOv8
│   ├── rec_module.py      # Logic của hệ thống tư vấn
│   └── utils.py           # Các hàm hỗ trợ (load config, xử lý format)
│
├── app.py                 # File chạy chính của giao diện Streamlit
├── requirements.txt       # Danh sách các thư viện cần cài đặt
├── .env.example           # Biến môi trường mẫu (nếu có)
├── CLAUDE.md              # File quy chuẩn coding & prompt cho AI Assistant
└── README.md              # Tài liệu mô tả dự án