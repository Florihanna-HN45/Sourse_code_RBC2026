# detection_dataset.py
# src/dataset/detection_dataset.py
import torch
from torch.utils.data import Dataset
# ----------------------------
# Dataset cho detection (bbox hộp)
from src.utils.augmentations import get_detection_transforms

train_tf, val_tf = get_detection_transforms(img_size=640)


class DetectionDataset(Dataset):
    def __init__(self, img_dir, ann_path, split_path, transforms=None):
        self.img_dir = img_dir
        self.ann_path = ann_path
        self.transforms = transforms
        self.samples = self._load_split(split_path)
        self.anns = self._load_annotations(ann_path)

    def _load_split(self, split_path):
        # TODO: load list file names for train/val
        return []

    def _load_annotations(self, ann_path):
        # TODO: parse COCO/VOC annotations -> dict: img_id -> boxes, labels(=1)
        return {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.transforms(image=image, bboxes=boxes, labels=labels)
        image, boxes, labels = sample["image"], sample["bboxes"], sample["labels"]

        img_id = self.samples[idx]
        image = self._load_image(img_id)  # TODO
        boxes, labels = self.anns[img_id]["boxes"], self.anns[img_id]["labels"]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        if self.transforms:
            image, target = self.transforms(image, target)  # TODO

        return image, target

    def _load_image(self, img_id):
        # TODO: read image from disk, convert to RGB
        return None