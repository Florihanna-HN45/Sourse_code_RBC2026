import os

# Định nghĩa cấu trúc thư mục
folders = [
    "cv-box-project/data/images",
    "cv-box-project/data/annotations",
    "cv-box-project/data/crops",
    "cv-box-project/data/splits",
    "cv-box-project/configs",
    "cv-box-project/src/dataset",
    "cv-box-project/src/models",
    "cv-box-project/src/train",
    "cv-box-project/src/infer",
    "cv-box-project/src/utils",
    "cv-box-project/src/export",
    "cv-box-project/logs",
    "cv-box-project/runs"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✅ Đã tạo xong cấu trúc thư mục cv-box-project")