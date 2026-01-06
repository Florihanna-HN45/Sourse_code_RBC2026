import os

# Định nghĩa cấu trúc thư mục và file
structure = {
    "cv-box-project/src/dataset": ["detection_dataset.py", "classification_dataset.py", "__init__.py"],
    "cv-box-project/src/models": ["mobilenetv2_backbone.py", "ssd_head.py", "multi_task_classifier.py", "__init__.py"],
    "cv-box-project/src/train": ["train_detector.py", "train_classifier.py", "__init__.py"],
    "cv-box-project/src/infer": ["run_pipeline.py", "__init__.py"],
    "cv-box-project/src/utils": ["transforms.py", "box_ops.py", "viz.py", "__init__.py"],
    "cv-box-project/src/export": ["to_onnx.py", "__init__.py"],
}

for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for f in files:
        path = os.path.join(folder, f)
        if not os.path.exists(path):
            with open(path, "w") as fp:
                fp.write("# " + f + "\n")
print("✅ Đã tạo xong các file .py rỗng")
# Tạo các thư mục trống không có file