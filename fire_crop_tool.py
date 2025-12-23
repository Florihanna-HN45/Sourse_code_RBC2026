import cv2
import numpy as np
import os

# --- CẤU HÌNH ---
INPUT_FOLDER = r"D:\1. STUDY\ROBOCON 2026\0. THAY LAM\dataset_raw_images" 
OUTPUT_FOLDER = r"D:\1. STUDY\ROBOCON 2026\0. THAY LAM\dataset_crops_wide" # Đổi tên folder output cho dễ phân biệt

# Tạo thư mục đầu ra
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def crop_fire_wide():
    files = os.listdir(INPUT_FOLDER)
    count = 0
    print(f"Đang xử lý {len(files)} ảnh...")

    for file_name in files:
        if not file_name.endswith((".jpg", ".png", ".jpeg")):
            continue
            
        img_path = os.path.join(INPUT_FOLDER, file_name)
        img = cv2.imread(img_path)
        if img is None: continue

        img_h, img_w = img.shape[:2]

        # 1. Lọc màu Lửa (HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_fire = np.array([0, 100, 200]) 
        upper_fire = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_fire, upper_fire)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 2. Tìm viền
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500: continue # Bỏ nhiễu
                
            x, y, w, h = cv2.boundingRect(cnt)

            # --- PHẦN CHỈNH SỬA QUAN TRỌNG Ở ĐÂY ---
            # Tính padding bằng 15% chiều rộng của bức ảnh gốc
            # Cậu có thể tăng/giảm số 0.15 này để khung rộng/hẹp hơn
            pad_ratio = 0.15 
            pad_x = int(img_w * pad_ratio)
            pad_y = int(img_h * pad_ratio) # Hoặc dùng pad_x cho cả 2 để khung vuông

            # Tính tọa độ mới và kẹp (clamp) để không tràn ra ngoài ảnh
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_w, x + w + pad_x)
            y2 = min(img_h, y + h + pad_y)

            # Cắt và lưu
            crop_img = img[y1:y2, x1:x2]
            
            # Chỉ lưu nếu ảnh cắt ra đủ lớn (tránh ảnh lỗi)
            if crop_img.size > 0:
                count += 1
                save_name = f"fire_wide_{count}.jpg"
                save_path = os.path.join(OUTPUT_FOLDER, save_name)
                cv2.imwrite(save_path, crop_img)
                # print(f"Đã lưu: {save_name}")

    print(f"✅ Xong! Đã cắt được {count} ảnh góc rộng.")
    print(f"Kiểm tra tại: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    crop_fire_wide()