import cv2

# --- CẤU HÌNH ---
dir = r'D:/1. STUDY/ROBOCON 2026/0. THAY LAM'
VIDEO_PATH = dir + "/output.mp4"  # Thay tên video của cậu vào

# Biến lưu toạ độ tạm
ix, iy = -1, -1
drawing = False
roi_coordinates = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, roi_coordinates

    # Nhấn chuột xuống -> Bắt đầu vẽ
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # Nhả chuột ra -> Kết thúc vẽ -> In toạ độ
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w = abs(x - ix)
        h = abs(y - iy)
        # Tính x, y góc trên bên trái chuẩn xác
        start_x = min(ix, x)
        start_y = min(iy, y)
        
        print(f"\n🎯 TOẠ ĐỘ TÌM ĐƯỢC: [x={start_x}, y={start_y}, w={w}, h={h}]")
        print(f"👉 Copy dòng này vào code cắt: box = [{start_x}, {start_y}, {w}, {h}]")
        
        # Vẽ hình chữ nhật lên hình để xác nhận
        cv2.rectangle(img, (start_x, start_y), (start_x + w, start_y + h), (0, 255, 0), 2)
        cv2.imshow('Video_Player', img)

cap = cv2.VideoCapture(VIDEO_PATH)

print("--- HƯỚNG DẪN ---")
print("1. Nhấn 'SPACE' (phím cách) để tạm dừng video tại cảnh cần lấy.")
print("2. Dùng chuột KÉO THẢ một vùng hình chữ nhật trên video.")
print("3. Nhìn vào Terminal để lấy toạ độ.")
print("4. Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    img = frame.copy()
    cv2.imshow('Video_Player', img)
    
    # Thiết lập hàm chuột cho cửa sổ này
    cv2.setMouseCallback('Video_Player', draw_rectangle)

    k = cv2.waitKey(30) & 0xFF
    if k == ord('q'): # Thoát
        break
    elif k == 32: # Phím Space: Tạm dừng để vẽ
        print("⏸ Đã tạm dừng. Hãy vẽ hình chữ nhật...")
        cv2.waitKey(0) # Chờ bấm phím bất kỳ để chạy tiếp

cap.release()
cv2.destroyAllWindows()