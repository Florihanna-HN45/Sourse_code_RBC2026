# ssd_head.py
# src/models/ssd_head.py (lớp box head cho SSD)
import torch
import torch.nn as nn

class SSDHead(nn.Module):
    def __init__(self, in_channels, num_anchors=6, num_classes=2):
        super().__init__()
        # num_classes = 2 (background + box)
        # TODO: define extra feature maps for SSD
        self.cls_heads = nn.ModuleList([])
        self.box_heads = nn.ModuleList([])
        # TODO: build heads for each feature map

    def forward(self, features):
        # features: list of feature maps from backbone + extras
        cls_logits = []
        box_reg = []
        # TODO: loop and compute heads
        return cls_logits, box_reg