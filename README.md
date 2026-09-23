# 🔥 Real-Time Fire & Smoke Detection with YOLO

An end-to-end Computer Vision pipeline designed for early wildfire, structure fire, and smoke detection in real-world surveillance systems.

---

## 📌 Project Overview
Early detection of fire and hazardous smoke is critical for minimizing environmental and property loss. This project delivers a high-accuracy, low-latency object detection model trained on custom annotated data to identify fire origins and rising smoke columns under varying lighting and atmospheric conditions.

---

## 🎬 Real-World Inferences (Testing Phase)

The trained model was benchmarked against unseen YouTube video sequences to test robust generalization against smoke diffusion, flickering flame artifacts, and complex backgrounds.

| Fire & Smoke Detection Demo | Smoke Dispersion Demo |
| :---: | :---: |
| ![Inference 1](assets/demo1.gif) | ![Inference 2](assets/demo2.gif) |

---

## 📊 Training Performance & Evaluation Metrics

The model converged with high precision, demonstrating strong class separation between subtle smoke textures and ambient background noise.

### 1. Training Loss & Metric Curves
![Training Results](assets/metrics/results.png)

### 2. Confusion Matrix & Precision-Recall
| Normalized Confusion Matrix | Precision-Recall (PR) Curve |
| :---: | :---: |
| ![Confusion Matrix](assets/metrics/confusion_matrix_normalized.png) | ![PR Curve](assets/metrics/BoxPR_curve.png) |

### 3. Validation Sample Predictions
Ground truth vs. predicted bounding boxes on unseen validation batches:
![Validation Batch Prediction](assets/metrics/val_batch0_pred.jpg)

---

## 📁 Dataset Details
* **Source:** [Smoke and Fire Detection Dataset (YOLO)](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) on Kaggle
* **Target Classes:** `0: Fire`, `1: Smoke`
* **Format:** YOLO annotation (`<class_id> <x_center> <y_center> <width> <height>`)
* **Preprocessing & Augmentation:** HSV jittering, horizontal flip, mosaic augmentation, and multi-scale resizing.

---

## 📂 Repository Layout
```text
fire-and-smoke-detection-yolo/
├── assets/
│   ├── demo1.gif                  # Sample YouTube inference demo
│   ├── demo2.gif                  # Sample YouTube inference demo
│   └── metrics/                   # Loss curves, confusion matrix, PR curves
│       ├── results.png
│       ├── confusion_matrix_normalized.png
│       ├── BoxPR_curve.png
│       └── val_batch0_pred.jpg
├── notebooks/
│   ├── image_processing.ipynb     # Exploratory data analysis & image transforms
│   ├── yolo_training.ipynb        # Fine-tuning loop, hyperparameter config
│   ├── results.csv                # Raw training epoch logs
│   └── training_args.yaml         # Training arguments & hyperparameters
├── weights/
│   └── best.pt                    # Best checkpoint weights
├── requirements.txt               # Environment dependencies
└── README.md
