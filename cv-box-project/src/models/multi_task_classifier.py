# multi_task_classifier.py
# src/models/multi_task_classifier.py
import torch
import torch.nn as nn

class MultiTaskClassifier(nn.Module):
    def __init__(self, backbone, in_channels, num_rf=2, num_types_global=30):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.head_rf = nn.Linear(in_channels, num_rf)            # real/fake
        self.head_type = nn.Linear(in_channels, num_types_global) # 30 loại

    def forward(self, x):
        feats = self.backbone(x)            # TODO: forward through features
        # feats shape: (B, C, H, W)
        pooled = self.pool(feats)
        pooled = pooled.view(pooled.size(0), -1)
        rf_logits = self.head_rf(pooled)
        type_logits = self.head_type(pooled)
        return rf_logits, type_logits