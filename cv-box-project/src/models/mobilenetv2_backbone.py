# mobilenetv2_backbone.py
# src/models/mobilenetv2_backbone.py
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

def build_mobilenetv2_backbone(pretrained=True, freeze=False):
    model = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
    features = model.features  # Sequential
    out_channels = 1280  # last conv output channels

    if freeze:
        for p in features.parameters():
            p.requires_grad = False

    return features, out_channels