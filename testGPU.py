# import torch

# if torch.cuda.is_available():
#     print("CUDA is available. Device count:", torch.cuda.device_count())
#     for i in range(torch.cuda.device_count()):
#         print(f"Device {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("CUDA không khả dụng. Có thể đang dùng GPU Intel hoặc CPU.")

import subprocess
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Thông số ---
# LẤY ĐƯỜNG DẪN model "best.pt" TỪ KẾT QUẢ TRAIN TRƯỚC
MODEL_PATH = "D:/ThiGiacMayTinh/CV-main/runs/detect/train_tomato_leaf/weights/best.pt"
DATASET_PEST_YAML = r"C:\Users\User\OneDrive - Ho Chi Minh City University of Foreign Languages and Information Technology - HUFLIT\Desktop\test_sau_la_tomato\la_processed\Leaf Mold"
IMGSZ = 512 # Phải dùng đúng imgsz đã train

# --- Lệnh test YOLO ---
cmd = [
    "yolo",
    "detect",
    "predict",  # <-- Vẫn dùng mode "val"
    f"model={MODEL_PATH}",
    f"source={DATASET_PEST_YAML}",
    f"imgsz={IMGSZ}",
     # <-- THAM SỐ QUAN TRỌNG NHẤT
    "device=0", 
    "verbose=True",
    "name=test_1leaf" # Tên thư mục lưu kết quả test mới
]

print("Bắt đầu TEST mô hình trên tập test...")
subprocess.run(cmd, check=True)
print("Hoàn tất test!")