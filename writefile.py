import os

base = "cv-box-project/src"
##Lấy ví dụ thôi nhé!
files_content = {
    "dataset/detection_dataset.py": """# detection_dataset.py
class DetectionDataset:
    def __init__(self):
        pass
    # TODO: implement
""",
    "dataset/classification_dataset.py": """# classification_dataset.py
class ClassificationDataset:
    def __init__(self):
        pass
    # TODO: implement
""",
    "models/mobilenetv2_backbone.py": """# mobilenetv2_backbone.py
def build_mobilenetv2_backbone(pretrained=True, freeze=False):
    # TODO: implement
    pass
""",
    # thêm các file khác tương tự...
}

for rel_path, content in files_content.items():
    full_path = os.path.join(base, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)

print("✅ Đã ghi nội dung vào các file .py")