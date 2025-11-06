# 🧠 Deep Abdominal Organ Segmentation

Deep learning-based segmentation of abdominal organs (such as liver, pancreas, ureters, and vesicular glands) using **DeepLabV3**, **U-Net++**, and **SegFormer** architectures on the **Dresden Anatomy Dataset**.

---

## 📘 Overview

This repository contains the implementation and evaluation of deep learning models for **automatic segmentation of intra-abdominal organs** in laparoscopic surgery images.  
The study compares the performance of three state-of-the-art architectures — **DeepLabV3**, **U-Net++**, and **SegFormer** — on the **Dresden Abdominal Anatomy Dataset**.

👉 **This work specifically focuses on the segmentation of abdominal organs that are difficult to delineate and typically achieve lower accuracy scores in previous studies.**  
All models were trained and evaluated under the same experimental conditions and hyperparameters to enable a fair comparison between architectures.  
Additionally, a **separate dedicated experiment was conducted for pancreas segmentation**, given its challenging structure and low contrast in the dataset.  
Each model was trained using images at three different resolutions — **128×128**, **256×256**, and **512×512** — to assess the effect of image size on segmentation accuracy.

By providing accurate, pixel-level segmentation, this study aims to enhance anatomical understanding during minimally invasive surgical procedures.

---

## 🧩 Models Implemented

| Model | Framework | Notes |
|--------|------------|--------|
| **DeepLabV3** | PyTorch | Atrous convolution-based semantic segmentation |
| **U-Net++** | PyTorch | Nested skip connections for better localization |
| **SegFormer** | PyTorch (HuggingFace Transformers) | Transformer-based lightweight segmentation model |

---

## 📊 Dataset

- **Dataset**: [Dresden Abdominal Anatomy Dataset](https://zenodo.org/record/XXXX)  
- **Type**: Annotated laparoscopic images  
- **Resolutions**: 128×128, 256×256, 512×512  
- **Classes**: Liver, Pancreas, Ureters, Vesicular glands, Intestinal veins  
- **Usage**: Used according to the dataset license and data usage agreement.

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/mustafaaksu/deep-abdomen-segmentation.git
cd deep-abdomen-segmentation
pip install -r requirements.txt
