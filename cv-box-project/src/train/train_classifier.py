# train_classifier.py
# src/train/train_classifier.py
import torch
from torch.utils.data import DataLoader
from src.dataset.classification_dataset import ClassificationDataset
from src.models.mobilenetv2_backbone import build_mobilenetv2_backbone
from src.models.multi_task_classifier import MultiTaskClassifier

def train_classifier(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = ClassificationDataset(cfg["crop_dir"], cfg["split_train"], transforms=None)
    ds_val   = ClassificationDataset(cfg["crop_dir"], cfg["split_val"], transforms=None)

    dl_train = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=cfg["batch_size"], shuffle=False)

    backbone, out_ch = build_mobilenetv2_backbone(pretrained=True, freeze=False)
    model = MultiTaskClassifier(backbone, out_ch, num_rf=2, num_types_global=30).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    ce = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg["epochs"]):
        model.train()
        for images, targets in dl_train:
            images = images.to(device)
            label_rf = targets["label_rf"].to(device)
            type_global = targets["type_global"].to(device)

            rf_logits, type_logits = model(images)

            loss_rf = ce(rf_logits, label_rf)
            loss_type = ce(type_logits, type_global)  # TODO: mask 15 loại theo real/fake
            loss = cfg["alpha"] * loss_rf + cfg["beta"] * loss_type

            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            # TODO: compute val acc for rf and type
            pass

        # TODO: save checkpoint