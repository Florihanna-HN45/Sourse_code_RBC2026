#DATA AUGMENTATIONS FUNCTIONS
# augmentations.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ----------------------------
# Augmentation cho classification (crop hộp)
# ----------------------------
def get_classification_transforms(img_size=224):
    train_tf = A.Compose([
        A.Resize(img_size+32, img_size+32),
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=0, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.JpegCompression(quality_lower=60, quality_upper=95, p=0.3),
        A.CoarseDropout(max_holes=2, max_height=32, max_width=32, p=0.2),
        ToTensorV2()
    ])

    val_tf = A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2()
    ])

    return train_tf, val_tf


# ----------------------------
# Augmentation cho detection (bbox hộp)
# ----------------------------
def get_detection_transforms(img_size=640):
    train_tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9,1.1), rotate=(-5,5), translate_percent=(0.0,0.02), p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.15),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.25),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_area=16,
        min_visibility=0.3
    ))

    val_tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))

    return train_tf, val_tf


# ----------------------------
# MixUp / CutMix cho classification
# ----------------------------
import torch
import random

def mixup_data(x, y, alpha=1.0):
    '''MixUp augmentation'''
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    '''CutMix augmentation'''
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size)

    # random bbox
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    w = int(W * (1 - lam)**0.5)
    h = int(H * (1 - lam)**0.5)
    x1 = max(cx - w // 2, 0)
    y1 = max(cy - h // 2, 0)
    x2 = min(cx + w // 2, W)
    y2 = min(cy + h // 2, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam