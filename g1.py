import cv2
import os
from ultralytics import YOLO

# --- CẤU HÌNH ---
VIDEO_PATH = r"D:/1. STUDY/ROBOCON 2026/0. THẦY LÂM/output_bat1den.mp4"  # <-- Điền tên video của cậu vào đây
MODEL_PATH = "fire_model.pt"
OUTPUT_FOLDER = r"D:/1. STUDY/ROBOCON 2026/0. THẦY LÂM/fire_data_crop"  # Tên thư mục sẽ lưu ảnh cắt ra

# Tạo thư mục lưu ảnh nếu chưa có
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load model
model = YOLO(MODEL_PATH)
class_names = model.names

def crop_and_save(frame, box_coords, count, cls_name):
    """
    Hàm cắt ảnh và lưu vào thư mục
    box_coords: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box_coords
    
    # --- XỬ LÝ CẮT ẢNH (QUAN TRỌNG) ---
    # OpenCV cắt ảnh theo thứ tự: frame[y_start : y_end, x_start : x_end]
    # Lưu ý phải kẹp giá trị max/min để không bị lỗi tràn khung hình
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Thực hiện cắt
    crop_img = frame[y1:y2, x1:x2]
    
    # Kiểm tra nếu ảnh cắt ra hợp lệ (không bị rỗng)
    if crop_img.size > 0:
        filename = f"{OUTPUT_FOLDER}/{cls_name}_{count}.jpg"
        cv2.imwrite(filename, crop_img)
        # print(f"Đã lưu: {filename}")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Kiểm tra xem video có mở được không
    if not cap.isOpened():
        print(f"Lỗi: Không tìm thấy video tại {video_path}")
        return

    crop_count = 0 # Biến đếm để đặt tên file ảnh
    thres = 0.5    # Ngưỡng tin cậy

    print("Đang xử lý video... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hết video.")
            break

        # Resize nhẹ nếu video 4K quá nặng, nếu video thường thì bỏ qua dòng này
        # frame = cv2.resize(frame, (640, 480))

        # --- DETECT ---
        results = model(frame, stream=True, verbose=False) # verbose=False để đỡ spam terminal

        is_warning = False # Cờ đánh dấu frame này có lửa không

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Lấy tọa độ
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                current_class = class_names[cls]

                if conf > thres:
                    is_warning = True
                    
                    # 1. Vẽ khung chữ nhật lên ảnh gốc (để hiển thị)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{current_class} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # 2. CẮT VÀ LƯU ẢNH (Task của cậu)
                    # Mỗi lần detect được sẽ lưu ảnh ra folder
                    crop_count += 1
                    crop_and_save(frame, [x1, y1, x2, y2], crop_count, current_class)

        # Hiển thị video
        cv2.imshow("Fire Detection & Crop", frame)

        # Chờ 1ms, nếu bấm 'q' thì thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(VIDEO_PATH)