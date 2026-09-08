# **Tài liệu Kỹ thuật Dự án AI: Hệ thống Nhận diện và Tư vấn Ô tô**

## **1\. Tổng quan Dự án**

Dự án xây dựng một hệ thống Trí tuệ Nhân tạo đa nhiệm, tích hợp khả năng nhận diện chính xác các dòng xe ô tô qua hình ảnh và tư vấn dòng xe phù hợp dựa trên nhu cầu cá nhân hóa của người dùng. Hệ thống được thiết kế đặc thù để tối ưu hóa hiệu suất khi vận hành tại môi trường cục bộ (Local Environment) thông qua định dạng ONNX.

## **2\. Ngôn ngữ Lập trình & Công nghệ Cốt lõi**

| Hạng mục | Công nghệ / Ngôn ngữ | Mục đích sử dụng   |
| :---- | :---- | :---- |
| Ngôn ngữ chính | Python 3.9+ | Phát triển toàn bộ logic AI, luồng dữ liệu và API xử lý hình ảnh. |
| Thị giác máy tính (CV) | YOLOv8, OpenCV, PyTorch | Phát hiện đối tượng (Object Detection), trích xuất đặc trưng và tiền xử lý ảnh đầu vào. |
| Định dạng & Tối ưu | ONNX, ONNX Runtime | Xuất mô hình để chạy độc lập không phụ thuộc framework gốc, tăng tốc độ suy luận (inference). |
| Giao diện người dùng | Streamlit hoặc Gradio | Xây dựng web-app nhanh chóng để người dùng tải ảnh và nhập nhu cầu tư vấn. |

## **3\. Kiến trúc Công nghệ Chi tiết**

### **3.1. Module Nhận diện xe (Computer Vision với YOLOv8)**

* **Cấu trúc mô hình:** Lựa chọn YOLOv8 (phiên bản YOLOv8n hoặc YOLOv8s) nhằm cân bằng giữa độ chính xác và tốc độ khung hình (FPS) trên cấu hình máy tính cá nhân.  
* **Quy trình huấn luyện:** Sử dụng framework PyTorch để tinh chỉnh (Fine-tune) mô hình YOLOv8 trên tập dữ liệu xe ô tô (như VMMRdb hoặc Stanford Cars). Kết quả quá trình huấn luyện xuất ra file trọng số .pt.  
* **Chuyển đổi sang ONNX:** Sử dụng bộ công cụ Ultralytics để thực hiện export.  
  from ultralytics import YOLO

  \# Tải trọng số mô hình đã huấn luyện  
  model \= YOLO('best.pt')

  \# Xuất mô hình sang định dạng ONNX  
  model.export(format='onnx', dynamic=True, simplify=True)  
      
* **Suy luận (Inference):** Khởi tạo InferenceSession từ onnxruntime để nạp mô hình best.onnx vào bộ nhớ và dự đoán trực tiếp từ ma trận ảnh của OpenCV.

### **3.2. Module Tư vấn (Recommendation Engine)**

* **Thuật toán:** Triển khai Content-based Filtering hoặc sử dụng mô hình K-Nearest Neighbors (KNN) từ thư viện Scikit-learn.  
* **Dữ liệu đầu vào:** Cấu trúc hóa các trường thông số (Giá bán, Phân khúc, Số chỗ ngồi, Mức tiêu hao nhiên liệu) vào DataFrame của Pandas.  
* **Đóng gói:** Chuyển đổi pipeline của Scikit-learn sang định dạng ONNX bằng thư viện skl2onnx để đồng bộ hoàn toàn kiến trúc chạy local.

## **4\. Yêu cầu Cài đặt Môi trường**

1. Thiết lập môi trường ảo Python (Virtual Environment) để cô lập thư viện.  
2. Cài đặt các gói phụ thuộc cơ bản (Requirements): onnxruntime, opencv-python, numpy, pandas, streamlit.  
3. Cấu hình Execution Provider trong ONNX Runtime thành CPUExecutionProvider (hoặc bổ sung CUDAExecutionProvider nếu phần cứng hỗ trợ card đồ họa rời).