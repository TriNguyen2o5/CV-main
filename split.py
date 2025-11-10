import os
import shutil
import random

# Thư mục gốc dataset leaf
BASE_DIR = "yolo_leaf"

# Tỷ lệ chia dataset
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1  # còn lại

# Thư mục con
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")

for subset in ["train", "val", "test"]:
    for sub in ["images", "labels"]:
        folder = os.path.join(BASE_DIR, subset, sub)
        os.makedirs(folder, exist_ok=True)
        # Xóa file cũ nếu có
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

# Lấy danh sách ảnh
imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png"))]
random.shuffle(imgs)
n = len(imgs)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

for i, img_file in enumerate(imgs):
    if i < train_end:
        subset = "train"
    elif i < val_end:
        subset = "val"
    else:
        subset = "test"

    # Copy ảnh
    shutil.copy(os.path.join(IMG_DIR, img_file), os.path.join(BASE_DIR, subset, "images", img_file))

    # Copy label nếu tồn tại
    lbl_file = img_file.rsplit(".", 1)[0] + ".txt"
    src_lbl = os.path.join(LBL_DIR, lbl_file)
    if os.path.exists(src_lbl):
        shutil.copy(src_lbl, os.path.join(BASE_DIR, subset, "labels", lbl_file))

print("✅ Chia dataset yolo_leaf thành công!")
print(f"- Train: {train_end} ảnh")
print(f"- Val: {val_end - train_end} ảnh")
print(f"- Test: {n - val_end} ảnh")
