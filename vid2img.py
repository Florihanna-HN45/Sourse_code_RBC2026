import cv2
import os

# --- CẤU HÌNH ---
dir = r'D:/1. STUDY/ROBOCON 2026/0. THAY LAM'
VIDEO_PATH = dir + "/output1.mp4"   # Tên file video của cậu
OUTPUT_FOLDER = dir + "/output1crop" # Thư mục chứa ảnh đầu ra
STEP = 30  # Quan trọng: Cứ 30 frame thì lưu 1 ảnh (để tránh trùng lặp)

# 1. Tạo thư mục nếu chưa có
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Đã tạo thư mục: {OUTPUT_FOLDER}")

# 2. Đọc video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Lỗi: Không mở được video!")
    exit()

frame_count = 0
saved_count = 0

print("Bắt đầu trích xuất... Nhấn 'q' để dừng sớm.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Đã hết video.")
        break
    
    # 3. Logic lưu ảnh: Chỉ lưu khi frame_count chia hết cho STEP
    if frame_count % STEP == 0:
        # Tạo tên file: frame_0.jpg, frame_1D0.jpg...
        filename = f"{OUTPUT_FOLDER}/frame_{frame_count}.jpg"
        
        # Lưu ảnh
        cv2.imwrite(filename, frame)
        saved_count += 1
        print(f"Đã lưu: {filename}")

    frame_count += 1
    
    # (Tùy chọn) Hiển thị video đang chạy để cậu biết nó đến đâu rồi
    cv2.imshow('Extracting...', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"--- HOÀN TẤT ---")
print(f"Tổng số ảnh đã lấy được: {saved_count} tấm")
print(f"Lưu tại: {os.path.abspath(OUTPUT_FOLDER)}")