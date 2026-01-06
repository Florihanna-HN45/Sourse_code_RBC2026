project_root/
├─ data/
│  ├─ images/                 # ảnh thô
│  ├─ annotations/            # nhãn detection (COCO/VOC)
│  ├─ crops/                  # crop hộp cho classification
│  └─ splits/                 # train/val/test lists
├─ configs/
│  ├─ detector.yaml
│  └─ classifier.yaml
├─ src/
│  ├─ dataset/
│  │  ├─ detection_dataset.py
│  │  └─ classification_dataset.py
│  ├─ models/
│  │  ├─ mobilenetv2_backbone.py
│  │  ├─ ssd_head.py
│  │  └─ multi_task_classifier.py
│  ├─ train/
│  │  ├─ train_detector.py
│  │  └─ train_classifier.py
│  ├─ infer/
│  │  └─ run_pipeline.py
│  ├─ utils/
│  │  ├─ transforms.py
│  │  ├─ box_ops.py
│  │  └─ viz.py
│  └─ export/
│     └─ to_onnx.py
├─ logs/
├─ runs/
└─ README.md
