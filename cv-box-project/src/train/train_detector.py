# train_detector.py
# src/train/train_detector.py
import torch
from torch.utils.data import DataLoader
from src.dataset.detection_dataset import DetectionDataset
from src.models.mobilenetv2_backbone import build_mobilenetv2_backbone
from src.models.ssd_head import SSDHead

def train_detector(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = DetectionDataset(cfg["img_dir"], cfg["ann_train"], cfg["split_train"], transforms=None)
    ds_val = DetectionDataset(cfg["img_dir"], cfg["ann_val"], cfg["split_val"], transforms=None)
    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True, collate_fn=lambda x: x)
    dl_val   = DataLoader(ds_val,   batch_size=cfg["batch_size"], shuffle=False, collate_fn=lambda x: x)

    backbone, out_ch = build_mobilenetv2_backbone(pretrained=True, freeze=False)
    detector = build_ssd_model(backbone, out_ch, num_classes=2)  # TODO: implement
    detector.to(device)

    optimizer = torch.optim.SGD(detector.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["step_size"], gamma=0.1)

    for epoch in range(cfg["epochs"]):
        detector.train()
        for batch in dl_train:
            # TODO: images, targets = preprocess
            # TODO: loss = detector_loss(...)
            optimizer.zero_grad()
            # TODO: loss.backward()
            optimizer.step()

        detector.eval()
        with torch.no_grad():
            for batch in dl_val:
                # TODO: compute val metrics
                pass

        scheduler.step()
        # TODO: save checkpoint

def build_ssd_model(backbone, out_ch, num_classes):
    # TODO: add extra layers, anchor generator, loss fn (cls + smooth L1)
    return None