# classification_dataset.py
# src/dataset/classification_dataset.py
import torch
from torch.utils.data import Dataset
# ----------------------------
# Dataset cho classification
# ----------------------------
from src.utils.augmentations import get_classification_transforms

train_tf, val_tf = get_classification_transforms(img_size=224)


class ClassificationDataset(Dataset):
    def __init__(self, crop_dir, split_path, transforms=None):
        self.crop_dir = crop_dir
        self.transforms = transforms
        self.samples = self._load_split(split_path)

    def _load_split(self, split_path):
        # TODO: load CSV/list: path, label_rf (0/1), label_type (0-14), type_global(0-29)
        return []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = self._load_image(self.samples[idx]["path"])
        if self.transforms:
            image = self.transforms(image=image)["image"]

        sample = self.samples[idx]
        image = self._load_image(sample["path"])  # TODO
        label_rf = sample["label_rf"]
        type_global = sample["type_global"]

        if self.transforms:
            image = self.transforms(image)  # TODO

        return image, {
            "label_rf": torch.tensor(label_rf, dtype=torch.long),
            "type_global": torch.tensor(type_global, dtype=torch.long)
        }

    def _load_image(self, path):
        # TODO: read image, RGB, to tensor
        return None