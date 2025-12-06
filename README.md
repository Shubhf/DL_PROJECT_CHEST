
# Pneumonia Detection from Chest X-Rays  
### ConvNeXt-Tiny & EfficientNet-B2 for Binary Classification (RSNA Dataset)

This project builds an end-to-end deep learning pipeline for pneumonia detection using chest X-ray images from the **RSNA Pneumonia Detection Challenge**.  
It converts DICOM files into PNGs, maps patient-level labels, trains two modern vision architectures, and evaluates them using clinical metrics like ROC-AUC and F1.

The repository contains:

- A complete training & evaluation notebook (`rsna_pneumonia.ipynb`)
- Trained ConvNeXt-Tiny and EfficientNet-B2 model checkpoints  
- A clean inference workflow  
- Visualizations: ROC curve, confusion matrix, metrics  

---

## 🧠 Project Overview

Radiographic pneumonia diagnosis is time-consuming and requires experienced radiologists.  
This project trains models to classify each X-ray as:

- **1 — Pneumonia detected**  
- **0 — Normal**

The focus is on:

- Reliable preprocessing from DICOM  
- Patient-ID–aligned label generation  
- Training ConvNeXt & EfficientNet on PNG images  
- Handling class imbalance with `pos_weight`  
- Robust evaluation with ROC-AUC, F1, and confusion matrix  

---

## 📁 Dataset Structure

The project expects the RSNA dataset organized as:
````
rsna-pneumonia-detection-challenge/
│
├── stage_2_train_labels.csv
└── rsna_yolo/
├── images/
│ ├── train/ # training PNGs
│ └── val/ # validation PNGs
└── labels/ # (optional) YOLO-format labels
````

Label creation follows:

- For each `patientId`, if **any bounding box exists**, label = `1`
- Otherwise, label = `0`

---

## 🏗️ Model Architectures

Two high-performance backbones were trained:

### **1. ConvNeXt-Tiny**
A modern hierarchical ConvNet inspired by Swin/ViT architectures.

### **2. EfficientNet-B2**
A scaled, parameter-efficient CNN known for strong medical imaging performance.

Both are:

- **ImageNet pretrained**
- Modified to output **one logit** (`num_classes=1`)
- Trained using **BCEWithLogitsLoss(pos_weight)`**  
  to handle imbalance effectively.

---

## 🔗 Pretrained Weights

You can download the trained `.pth` files here:

| Model               | Download |
|--------------------|----------|
| **EfficientNet-B2** | [Google Drive](https://drive.google.com/file/d/1cQ9Gw_DiERBEWTO7FQxu70zB7RILzsVV/view?usp=sharing) |
| **ConvNeXt-Tiny**   | [Google Drive](https://drive.google.com/file/d/1SPc_Gx6-S9GFLQmeYwa7mZbFbFF7LVSS/view?usp=sharing) |

Place them in the project directory to run evaluation or inference.

---

## 🚀 Training the Models
rsna_pneumonia.ipynb

Steps include:

- loading & preprocessing PNGs  
- balanced dataloaders  
- ConvNeXt-Tiny training  
- EfficientNet-B2 training  
- best-checkpoint saving (`best_model.pth`)  
- ROC-AUC monitoring  
- F1 threshold selection  

You can modify `EPOCHS`, `LR`, and model choice directly in the notebook.

---

## 📈 Evaluation Results

After training, the notebook computes:

- **ROC-AUC**
- **Precision, Recall, F1**
- **Normalized Confusion Matrix**
- **Best threshold for F1**

🙌 Acknowledgements

Dataset:
RSNA Pneumonia Detection Challenge
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

Model backbones from timm by Ross Wightman.







