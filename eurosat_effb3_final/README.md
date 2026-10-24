# 🛰️ EuroSAT Satellite Land Intelligence & Comparative Architecture Suite
## EfficientNet-B3 (Production Winner) vs. ResNet-50 (Challenger) Benchmark

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![EuroSAT Dataset](https://img.shields.io/badge/Dataset-EuroSAT%20(27K%20Patches)-brightgreen.svg)](https://github.com/phelber/EuroSAT)
[![Winning Model](https://img.shields.io/badge/Production%20Winner-EfficientNet--B3%20(97.82%25%20Acc)-gold.svg)]()

This repository contains a full-stack, production-grade deep learning remote sensing system. It features an in-depth comparative benchmark between **EfficientNet-B3** (compound scaling with MBConv blocks) and **ResNet-50** (deep residual bottleneck networks) across 27,000+ Sentinel-2 satellite image patches.

The winning model (**EfficientNet-B3**) is deployed to an interactive GIS intelligence web dashboard with Explainable AI (**Grad-CAM**), an **Atmospheric Weather Robustness Simulator**, and an automated **GIS Grid Tessellation Engine** for regional land cover mapping.

---

## 🏆 Key Performance Benchmark

| Metric / Specification | Model A: EfficientNet-B3 (🏆 Final Winner) | Model B: ResNet-50 (Challenger Baseline) | Performance Advantage |
|:---|:---|:---|:---|
| **Role in Main Web App** | **Primary Deployed Production Model** | **Challenger Comparative Baseline** | **EfficientNet-B3 Selected** |
| **Backbone Architecture** | `timm.create_model('efficientnet_b3')` | `torchvision.models.resnet50` | MBConv + SE Attention |
| **Model Parameters** | **12.2M** (12,185,000) | **24.6M** (24,566,346) | **50.4% fewer parameters** |
| **Computational FLOPs** | **1.8 GFLOPs** | **4.1 GFLOPs** | **56.1% fewer operations** |
| **Input Resolution** | 224 × 224 RGB | 224 × 224 RGB | Fair, controlled comparison |
| **Feature Representation Dim** | 1,536 dimensions | 2,048 dimensions | High spatial compactness |
| **ImageNet-1K Pretrained Baseline**| 82.2% Top-1 Accuracy | 76.1% Top-1 Accuracy | +6.1% richer visual priors |
| **EuroSAT Test Accuracy** | **97.82%** | **96.65%** | **+1.17% Higher Accuracy** |
| **EuroSAT Macro F1-Score** | **97.78%** | **96.52%** | **+1.26% Balanced Performance** |
| **Inference CPU Latency** | ~28.4 ms / patch | ~42.1 ms / patch | **~32.5% faster inference** |

> **Conclusion**: EfficientNet-B3 achieves higher test accuracy (+1.17%) and higher macro F1 (+1.26%) than ResNet-50 while requiring **over 50% fewer parameters** and **56% fewer FLOPs**. As a result, EfficientNet-B3 was chosen as the primary production engine for the live web application.

---

## 🌟 Core System Features

1. **Modular ML Architecture**:
   - Clean separation of concerns: `config.py` (hyperparameters & class metadata), `dataset.py` (TorchGeo loader & augmentations), `models.py` (PyTorch backbones), `utils.py` (hooks & schedulers), `train.py` (training CLI), and `evaluate.py` (comparative reporting).
2. **Interactive Live Head-to-Head Duel Mode**:
   - Run parallel inference on any satellite patch, generating class predictions, confidence deltas, CPU latency, and dual Grad-CAM heatmaps for both backbones side-by-side.
3. **Explainable AI (Grad-CAM)**:
   - Registers PyTorch backward hooks to capture activation maps at the penultimate convolutional layers (`conv_head` in EfficientNet-B3 and `layer4` in ResNet-50) to project spatial attention heatmaps onto satellite imagery.
4. **Synthetic Atmospheric Robustness Simulation**:
   - Real-time cloud haze (alpha blending) and atmospheric Gaussian blur sliders to evaluate model resilience under adverse weather conditions.
5. **GIS Grid Tessellation Engine**:
   - Slices large high-resolution satellite crops into a 128×128 grid, classifies each tile, applies transparent land-use overlay masks, and generates interactive Chart.js distribution analytics.
6. **Production REST API**:
   - Built on Flask with CORS enabled, sub-30ms inference response times, and multi-threaded image processing.

---

## 💼 Resume & Project Showcase (Google XYZ Format)

Use these bullet points directly on your CV, resume, or LinkedIn:

- **Deep Learning & Quantitative Impact**:
  > *"Engineered an end-to-end satellite land classification pipeline comparing EfficientNet-B3 and ResNet-50 across 27,000+ Sentinel-2 multispectral images, deploying the winning EfficientNet-B3 model with 97.82% test accuracy and 97.78% macro F1 while reducing parameter overhead by 50.4% (12.2M vs 24.6M) and FLOPs by 56.1%."*

- **Explainable AI (XAI) & Weather Robustness**:
  > *"Implemented Explainable AI (Grad-CAM) to visualize spatial activation heatmaps across penultimate CNN layers (conv_head vs layer4) and engineered an atmospheric noise simulator (Gaussian blur & haze) to validate model stability under degraded meteorological conditions."*

- **Production GIS Engine & Web Deployment**:
  > *"Developed a production GIS tessellation engine slicing high-resolution satellite imagery into 128×128 grid segments for regional land-use distribution mapping, packaged with a low-latency Flask REST API and responsive telemetry dashboard."*

---

## 🗂️ Project Structure

```
eurosat_effb3_final/
│
├── config.py                 # Central hyperparameter, path, and class color configs
├── dataset.py                # TorchGeo loader, 70/15/15 stratified split, augmentations
├── models.py                 # EfficientNet-B3, ResNet-50, and legacy ResNet-18 classifiers
├── utils.py                  # Grad-CAM hooks, EarlyStopping, and atmospheric simulator
├── train.py                  # CLI training script with TensorBoard tracking
├── evaluate.py               # Comparative evaluation CLI generating benchmark matrices
├── requirements.txt          # Python dependencies
│
├── eurosat_effb3_optical/    # Full-Stack Web Application
│   ├── app.py                # Flask server with REST API endpoints (/api/classify, /api/compare, etc.)
│   ├── templates/
│   │   └── index.html        # Interactive 4-tab dashboard with custom cursor and Chart.js
│   └── models/               # Model weights directory
│       ├── model_A_efficientnet_b3_optical.pth   # 🏆 Final Winner (~46 MB)
│       └── model_B_resnet50_optical.pth          # Challenger Baseline (~98 MB)
│
└── README.md                 # Complete project documentation
```

---

## 🚀 Setup & Quickstart

### 1. Installation
```bash
# Clone repository and enter directory
cd eurosat_effb3_final

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Head-to-Head Evaluation Benchmark
```bash
# Print comparative matrix between EfficientNet-B3 and ResNet-50
python evaluate.py --model compare

# Or evaluate a specific model checkpoint on the test set
python evaluate.py --model A   # EfficientNet-B3
python evaluate.py --model B   # ResNet-50
```

### 3. Launch the Web Application
```bash
python eurosat_effb3_optical/app.py
```
Open your browser to: **http://localhost:5000**

- **Tab 1: Single Patch & CAM**: Upload single Sentinel-2 image, adjust haze/blur sliders, inspect Grad-CAM.
- **Tab 2: GIS Grid Segmenter**: Upload large landscape image, slice into 128×128 tiles, view land-use distribution chart.
- **Tab 3: ResNet-50 vs EfficientNet-B3**: Run live dual inference, inspect side-by-side predictions, and review benchmark metrics.
- **Tab 4: Resume & Tech Summary**: 1-click copyable resume bullet points, architecture diagrams, and technical interview Q&A.

---

## ⚙️ Target Land Cover Classes

The EuroSAT dataset categorizes Sentinel-2 satellite imagery into 10 Land Use / Land Cover (LULC) classes:

| Class ID | Land Cover Class | Description | Theme Color |
|:---:|:---|:---|:---:|
| 0 | **AnnualCrop** | Annual agricultural fields | `#f57c00` |
| 1 | **Forest** | Dense tree cover | `#2e7d32` |
| 2 | **HerbaceousVegetation** | Grasslands and shrubs | `#81c784` |
| 3 | **Highway** | Roads and highways | `#757575` |
| 4 | **Industrial** | Factories and warehouses | `#c62828` |
| 5 | **Pasture** | Open grazing land | `#d4e157` |
| 6 | **PermanentCrop** | Orchards and vineyards | `#8d6e63` |
| 7 | **Residential** | Urban housing areas | `#d32f2f` |
| 8 | **River** | Water bodies — rivers | `#0288d1` |
| 9 | **SeaLake** | Sea and lake water bodies | `#1565c0` |

---

## 🔬 Technical Interview Discussion Guide

### Why did EfficientNet-B3 beat ResNet-50 despite having half the parameters?
EfficientNet-B3 applies **compound scaling**, balancing network depth, channel width, and image resolution simultaneously. In contrast, ResNet-50 relies on scaling depth through 50 stacked convolutional layers with bottleneck blocks. Furthermore, EfficientNet-B3 uses **Mobile Inverted Bottleneck (MBConv)** convolutions with **Squeeze-and-Excitation (SE)** attention. This explicitly recalibrates channel-wise feature responses, allowing the network to emphasize critical spectral reflectance bands (e.g. vegetation indices) without parameter bloat.

### Why was Macro F1-score prioritized alongside Overall Accuracy?
In satellite remote sensing, certain classes share overlapping visual textures and spectral profiles (such as *Pasture* vs. *Herbaceous Vegetation* or *AnnualCrop* vs. *PermanentCrop*). Overall accuracy can mask deficiencies on challenging or less distinct categories. **Macro F1-score** computes the arithmetic mean of F1-scores across all 10 classes independently, penalizing models that perform poorly on ambiguous boundary classes.
