import cv2
import os
import time

# --- CẤU HÌNH ---
dir = r'D:/1. STUDY/ROBOCON 2026/0. THAY LAM'
VIDEO_PATH = dir + "/output_bat1den.mp4"         # Đường dẫn video
OUTPUT_FOLDER = dir + "/dataset_mouse_crop"     # Thư mục lưu ảnh

# Tạo thư mục
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Biến toàn cục
drawing = False
ix, iy = -1, -1
roi = None  # Lưu toạ độ vùng chọn: (x, y, w, h)
is_recording = False # Biến trạng thái: Có đang tự động cắt không?
crop_count = 0

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, roi

    # Nhấn chuột xuống -> Bắt đầu vẽ
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        roi = None # Reset vùng chọn cũ

    # Kéo chuột -> Cập nhật hiển thị (tùy chọn, ở đây ta xử lý lúc thả chuột thôi cho mượt)
    
    # Thả chuột ra -> Chốt vùng chọn
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w = abs(x - ix)
        h = abs(y - iy)
        # Lấy góc trên bên trái
        start_x = min(ix, x)
        start_y = min(iy, y)
        
        if w > 5 and h > 5: # Chỉ nhận nếu khung đủ lớn
            roi = (start_x, start_y, w, h)
            print(f"🎯 Đã chọn vùng: {roi}")

def save_crop(frame, box):
    global crop_count
    if box is None: return
    
    x, y, w, h = box
    # Kẹp toạ độ trong khung hình để không lỗi
    h_img, w_img = frame.shape[:2]
    x = max(0, min(x, w_img))
    y = max(0, min(y, h_img))
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    
    crop = frame[y:y+h, x:x+w]
    
    if crop.size > 0:
        crop_count += 1
        # Đặt tên theo thời gian thực để không bao giờ trùng
        filename = f"fire_{int(time.time())}_{crop_count}.jpg"
        path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(path, crop)
        return True
    return False

# --- MAIN ---
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("Tool Cat Data")
cv2.setMouseCallback("Tool Cat Data", mouse_callback)

paused = False
print("--- HƯỚNG DẪN SỬ DỤNG ---")
print("🖱  KÉO CHUỘT: Chọn vùng cần cắt.")
print("SPACE: Tạm dừng / Tiếp tục video.")
print("S    : Save (Lưu) ảnh crop hiện tại ngay lập tức.")
print("R    : Record (Bật/Tắt chế độ tự động cắt khi video chạy).")
print("C    : Clear (Xóa vùng chọn).")
print("Q    : Quit (Thoát).")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Hết video -> Quay lại từ đầu.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
    else:
        # Nếu đang pause thì giữ nguyên frame cũ để vẽ cho dễ
        pass

    if frame is None: break
    
    display_img = frame.copy()

    # 1. Vẽ khung vùng chọn (Nếu có)
    if roi:
        rx, ry, rw, rh = roi
        color = (0, 0, 255) if is_recording else (0, 255, 0) # Đỏ nếu đang quay, Xanh nếu đang chọn
        cv2.rectangle(display_img, (rx, ry), (rx+rw, ry+rh), color, 2)
        
        if is_recording:
            cv2.putText(display_img, "AUTO RECORDING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # TỰ ĐỘNG LƯU NẾU ĐANG CHẠY VIDEO
            if not paused:
                saved = save_crop(frame, roi)
                if saved: 
                    # Hiệu ứng nháy nhẹ
                    cv2.circle(display_img, (30, 30), 10, (0, 0, 255), -1)

    cv2.imshow("Tool Cat Data", display_img)

    # Xử lý phím bấm
    key = cv2.waitKey(120 if not paused else 100) & 0xFF

    if key == ord('q'): # Thoát
        break
        
    elif key == 32: # SPACE: Pause/Play
        paused = not paused
        print(f"{'Tạm dừng' if paused else 'Tiếp tục'}")
        
    elif key == ord('c'): # Clear
        roi = None
        is_recording = False
        print("Đã xóa vùng chọn.")
        
    elif key == ord('s'): # Save 1 tấm
        if roi:
            save_crop(frame, roi)
            print(f"📸 Đã chụp 1 tấm! (Tổng: {crop_count})")
        else:
            print("⚠️ Chưa chọn vùng nào cả!")
            
    elif key == ord('r'): # Toggle Record
        if roi:
            is_recording = not is_recording
            if is_recording: print("🔴 BẮT ĐẦU TỰ ĐỘNG CẮT...")
            else: print("⚪️ ĐÃ DỪNG CẮT.")
        else:
            print("⚠️ Hãy chọn vùng trước khi Record!")

cap.release()
cv2.destroyAllWindows()