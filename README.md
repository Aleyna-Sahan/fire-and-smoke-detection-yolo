# 🔥 Real-Time Fire & Smoke Detection with YOLO

An end-to-end computer vision project focused on early fire and smoke detection in real-world surveillance scenarios using the YOLO object detection architecture.

---

## 📌 Project Overview
Early detection of smoke and fire plays a crucial role in disaster mitigation and public safety. This project trains and evaluates a YOLO-based detector capable of identifying smoke plumes and open flame patterns under varying environmental conditions.

The trained model was subsequently evaluated against real-world YouTube video streams to test generalization capability against false positives (e.g., lighting variations, complex backgrounds).

---

## 🎬 Real-World Inferences (Test Results)

| Test Video 1 (Fire & Smoke) | Test Video 2 (Smoke Detection) |
| :---: | :---: |
| ![Test 1](assets/demo1.gif) | ![Test 2](assets/demo2.gif) |
*(Not: assets içine mp4 yerine 5-10 saniyelik gif koyarsanız GitHub doğrudan README üzerinde otomatik oynatır.)*

---

## 📊 Dataset
The model was trained on the **[Smoke and Fire Detection Dataset (YOLO)](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)** from Kaggle.
* **Classes:** `Fire`, `Smoke`
* **Annotations:** YOLO format (Bounding boxes with normalized coordinates)
* **Pre-processing:** Applied data augmentation (rotation, scaling, contrast adjustment) to handle edge scenarios.

---

## 🛠 Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch, Ultralytics YOLO
* **Computer Vision:** OpenCV
* **Environment:** Jupyter Notebook / Google Colab

---

## 📂 Repository Structure
```text
├── assets/                  # Demo gifs/images extracted from YouTube inference
├── notebooks/
│   ├── image_processing.ipynb   # Initial data analysis and visual pre-processing
│   └── yolo_training.ipynb      # Model training pipeline, hyperparameter tuning & loss curves
├── weights/                 # Trained model weights (best.pt)
├── requirements.txt         # Dependencies
└── README.md
