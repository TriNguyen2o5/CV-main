import numpy as np
import cv2
from PIL import Image

# Tạo bộ lọc CLAHE một lần duy nhất để tái sử dụng
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image_for_yolo(image_path_or_frame, apply_enhancements=False):
    """
    Hàm xử lý TỔNG QUÁT (cho cả Upload và Camera).
    
    1. Đọc ảnh (an toàn Unicode).
    2. (TÙY CHỌN) Áp dụng cải thiện (CLAHE + Khử nhiễu) nếu `apply_enhancements=True`.
    3. Trả về:
       - img_bgr_processed (Numpy array BGR): Ảnh đã xử lý (hoặc ảnh gốc) để đưa vào YOLO.
       - img_pil_display (PIL.Image RGB): Ảnh gốc (chưa xử lý) để hiển thị so sánh.
    """
    
    frame = None
    original_pil_image = None
    
    try:
        # --- Bước 1: Đọc Ảnh ---
        if isinstance(image_path_or_frame, str):
            # 1. Đọc bằng PIL (an toàn với Unicode/HEIC)
            original_pil_image = Image.open(image_path_or_frame).convert('RGB')
            # 2. Chuyển PIL (RGB) sang OpenCV (BGR)
            frame = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)
        else:
            # Đây là frame từ webcam (BGR), dùng trực tiếp
            frame = image_path_or_frame
            # Chuyển BGR sang RGB để tạo ảnh PIL gốc
            original_pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if frame is None:
            raise ValueError("Nguồn ảnh không hợp lệ")

        # --- Bước 2: Xử lý Cải thiện (Tùy chọn) ---
        if apply_enhancements:
            # 1. Cân bằng sáng (Auto-Fix)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # 2. Khử nhiễu nhẹ
            img_denoised = cv2.fastNlMeansDenoisingColored(img_enhanced, None, 5, 5, 7, 21)
            
            # Trả về ảnh ĐÃ XỬ LÝ
            return img_denoised, original_pil_image
        else:
            # Trả về ảnh GỐC (chưa xử lý)
            return frame, original_pil_image

    except Exception as e:
        print(f"Lỗi trong image_processor (YOLO): {e}")
        return None, None