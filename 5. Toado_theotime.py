import cv2
import os

# --- CẤU HÌNH NHIỆM VỤ (ĐIỀN TOẠ ĐỘ CẬU VỪA TÌM VÀO ĐÂY) ---
dir = r'D:/1. STUDY/ROBOCON 2026/0. THAY LAM'
VIDEO_PATH = dir + "/output.mp4"      
OUTPUT_BASE_FOLDER = dir + "/time6"

# Cấu trúc: [Start_Frame, End_Frame, [x, y, w, h], "Tên_Class"]
# Nếu muốn cắt cả video thì để End_Frame thật lớn (vd: 10000)
TASKS = [
    # Nhiệm vụ 1: Cắt đám cháy ở góc trái từ đầu đến frame 300
    {"start": 0, "end": 1000, "box": [489, 350, 496, 281]},
    
]

if not os.path.exists(OUTPUT_BASE_FOLDER):
    os.makedirs(OUTPUT_BASE_FOLDER)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
saved_count = 0

print(f"Đang thực hiện {len(TASKS)} nhiệm vụ cắt ảnh...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Duyệt qua các nhiệm vụ xem frame hiện tại có nằm trong vùng cần cắt không
    for task in TASKS:
        if task["start"] <= frame_count <= task["end"]:
            x, y, w, h = task["box"]
                        
            # --- CẮT ẢNH ---
            # Lưu ý padding nếu thích (đoạn này đang cắt đúng y xì toạ độ)
            crop = frame[y : y+h, x : x+w]
            
            if crop.size > 0:
                # Lưu ảnh
                filename = f"{frame_count}.jpg"
                path = os.path.join(OUTPUT_BASE_FOLDER, filename)
                cv2.imwrite(path, crop)
                saved_count += 1

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Đang xử lý frame {frame_count}...")

cap.release()
print(f"✅ Xong! Tổng cộng đã cắt {saved_count} ảnh.")