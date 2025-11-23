import subprocess
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Thông số
MODEL_DET = "yolov8m.pt"
DATASET_PEST_YAML = "D:/ThiGiacMayTinh/CV-main/Dataset with augmentation/data.yaml"
EPOCHS = 50
IMGSZ = 640
BATCH = 8

# Lệnh train YOLO leaf detection
cmd = [
    "yolo",
    "detect",
    "train",
    f"model={MODEL_DET}",
    f"data={DATASET_PEST_YAML}",
    f"epochs={EPOCHS}",
    f"imgsz={IMGSZ}",
    f"batch={BATCH}",
    "device=0", 
    "verbose=True",
    "name=train_tomato_leaf_v8m"
]

print("Bắt đầu train YOLO leaf detection...")
subprocess.run(cmd, check=True)
print("Hoàn tất train YOLO leaf detection!")
