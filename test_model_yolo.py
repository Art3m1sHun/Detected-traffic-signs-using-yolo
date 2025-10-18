import cv2
from ultralytics import YOLO
import time
from pathlib import Path
import os
import torch # Đã thêm thư viện torch

# --- Cấu hình ---
# ĐƯỜNG DẪN ĐẾN MÔ HÌNH ĐÃ HUẤN LUYỆN CỦA BẠN (VUI LÒNG CẬP NHẬT)
MODEL_PATH = r"D:\Python\yolo11n.pt" 
# Chọn nguồn video: 0 cho webcam mặc định, hoặc đường dẫn đến file video (.mp4)
VIDEO_SOURCE = 0 

# --- Khởi tạo ---
try:
    # Giải phóng bộ nhớ GPU trước khi tải mô hình
    if torch.cuda.is_available():
        torch.cuda.empty_cache() # GIẢI PHÓNG BỘ NHỚ CUDA
    
    # 1. Tải mô hình YOLO đã train
    print(f"Đang tải mô hình từ: {MODEL_PATH}")
    # Đặt mô hình chạy trên GPU nếu có thể
    device = 0 if torch.cuda.is_available() else 'cpu' 
    model = YOLO(MODEL_PATH)
    
    # 2. Khởi tạo webcam (hoặc video file)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise IOError(f"Không thể mở nguồn video: {VIDEO_SOURCE}")
        
    print("Mô hình đã tải. Sẵn sàng nhận diện thời gian thực.")
    
except Exception as e:
    print(f"LỖI KHỞI TẠO: Vui lòng kiểm tra đường dẫn mô hình và kết nối webcam.")
    print(f"Chi tiết lỗi: {e}")
    # Đảm bảo giải phóng tài nguyên nếu xảy ra lỗi khởi tạo
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    exit()

# Thiết lập FPS (Frames Per Second) và tính toán thời gian
prev_time = time.time()
fps_avg = 0

# --- Vòng lặp Nhận diện Thời gian Thực ---
while cap.isOpened():
    # Đọc frame từ camera
    ret, frame = cap.read()
    
    if not ret:
        print("Không thể đọc frame từ nguồn video. Kết thúc.")
        break

    # 1. Thực hiện Phát hiện (Inference)
    results = model.predict(
        source=frame, 
        stream=False, 
        verbose=False, 
        imgsz=128 # Giữ nguyên 320 để tiết kiệm bộ nhớ GPU
    )
    
    # 2. Lấy frame đã được vẽ bounding box và nhãn
    annotated_frame = results[0].plot()
    
    # 3. Tính toán và hiển thị FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Làm mịn FPS (chạy trung bình động đơn giản)
    fps_avg = 0.9 * fps_avg + 0.1 * fps if fps_avg else fps
    fps_text = f"FPS: {fps_avg:.2f}"
    
    # Vẽ FPS lên góc trên bên trái của frame
    cv2.putText(annotated_frame, fps_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 4. Hiển thị kết quả
    cv2.imshow("YOLO Real-time Detection", annotated_frame)
    
    # Thoát khỏi vòng lặp khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Dọn dẹp Tài nguyên ---
cap.release()
cv2.destroyAllWindows()
print("Đã đóng cửa sổ nhận diện.")
