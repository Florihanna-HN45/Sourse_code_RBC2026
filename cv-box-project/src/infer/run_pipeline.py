# run_pipeline.py
# src/infer/run_pipeline.py
import torch
from src.utils.box_ops import nms  # TODO
from src.utils.viz import draw  # TODO

def load_detector(path):
    # TODO: load SSD model weights
    return None

def load_classifier(path):
    # TODO: load multi-task classifier weights
    return None

def detect_boxes(detector, image, conf_th=0.5, iou_th=0.5):
    # TODO: run detector, get boxes + scores
    # boxes = ...
    # scores = ...
    # keep = nms(boxes, scores, iou_th)
    # boxes = boxes[keep][scores[keep] > conf_th]
    return []

def classify_box(classifier, crop):
    # TODO: run classifier, get rf_logits, type_logits
    # rf = argmax -> 0 real / 1 fake
    # type_id = argmax -> 0-29
    return {"rf": 0, "type_id": 0}

def run(image_path, det_ckpt, clf_ckpt):
    image = _load_image(image_path)  # TODO
    detector = load_detector(det_ckpt)
    classifier = load_classifier(clf_ckpt)
    boxes = detect_boxes(detector, image)

    results = []
    for b in boxes:
        crop = _crop(image, b)  # TODO
        pred = classify_box(classifier, crop)
        results.append({"box": b, **pred})

    # TODO: draw and return output format for robot
    return results