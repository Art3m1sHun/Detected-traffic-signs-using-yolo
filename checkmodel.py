from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import os
import torch
from IPython.display import display # Dùng để hiển thị ảnh trong môi trường Jupyter/Colab

# --- Cấu hình Đường dẫn (Vui lòng kiểm tra lại) ---
# NOTE: Các đường dẫn này được lấy từ mã gốc của bạn.
IMAGE_DIR = Path(r"D:\Python\traffic_sign\data_signal_yolo\train\images")
LABELS_DIR = Path(r"D:\Python\traffic_sign\data_signal_yolo\train\labels")
MODEL_PATH = r"D:\Python\runs\detect\train\weights\best.pt"

# Chuẩn bị
try:
    list_data_test = list(IMAGE_DIR.iterdir())
    LABELS_DIR.mkdir(exist_ok=True)
    model = YOLO(MODEL_PATH)
except Exception as e:
    # Xử lý lỗi nếu không tìm thấy thư mục hoặc model
    print(f"Lỗi khởi tạo: Vui lòng kiểm tra lại đường dẫn IMAGE_DIR, LABELS_DIR và MODEL_PATH. Chi tiết lỗi: {e}")
    exit()


# Xóa tệp kết quả cũ và tạo tệp mới
answer_path = Path("./answer.txt")
if answer_path.exists():
  os.remove(answer_path)
answer_path.touch(exist_ok=True)

# Khởi tạo biến để lưu ảnh cuối cùng đã được vẽ box
last_img_display = None 

# Thử tải font chữ để vẽ nhãn
try:
    font = ImageFont.truetype("arial.ttf", 20) 
except IOError:
    font = ImageFont.load_default()

# --- Vòng lặp Xử lý và Vẽ Box ---
for img_path in tqdm(list_data_test, desc="Đang xử lý ảnh"):
    try:
        # 1. Chạy phát hiện (Inference)
        result = model([img_path], verbose=False)[0] # verbose=False để giảm log đầu ra
        boxes = result.boxes
        
        # 2. Mở ảnh và chuẩn bị công cụ vẽ
        img = Image.open(img_path).convert("RGB") # Chuyển sang RGB để đảm bảo vẽ box hoạt động
        img_draw = ImageDraw.Draw(img)
        width, height = img.size
            
        # 3. Ghi kết quả vào tệp và vẽ hộp giới hạn
        with open(answer_path, "a") as file:
            for bbox in boxes:
                # Lấy tọa độ chuẩn hóa (YOLO format: x_center, y_center, width, height)
                x_norm, y_norm, w_norm, h_norm = bbox.xywhn[0].tolist()
                cls_id = int(bbox.cls[0].item())
                
                # Ghi kết quả vào tệp answer.txt
                file.write(f"{img_path.stem} {cls_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

                # Lấy tọa độ không chuẩn hóa (Pixel: x1, y1, x2, y2) để vẽ lên ảnh
                x1, y1, x2, y2 = bbox.xyxy[0].tolist()
                
                # Lấy tên lớp và độ tin cậy
                conf = bbox.conf[0].item()
                label_name = model.names.get(cls_id, f'Class {cls_id}')
                label_text = f"{label_name} ({conf:.2f})"

                # Vẽ hộp giới hạn (Bounding Box)
                img_draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
                
                # Vẽ nhãn
                text_x = x1
                # Đặt nhãn phía trên box nếu đủ không gian, nếu không đặt bên dưới
                text_y = y1 - 25 if y1 > 25 else y1 + 5 
                img_draw.text((text_x, text_y), label_text, fill="red", font=font)

        # Lưu tham chiếu đến ảnh cuối cùng đã được xử lý
        last_img_display = img 
        
        # HIỂN THỊ TỪNG ẢNH SAU KHI XỬ LÝ (đã bỏ comment)
        display(img)

    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {img_path.name}: {e}")
        continue


print("\n--- Hoàn thành Xử lý ---")
if last_img_display:
    print(f"Lưu ý: Ảnh cuối cùng đã được hiển thị phía trên. Đã xử lý {len(list_data_test)} ảnh.")
