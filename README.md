# 🛰️ EuroSAT Multimodal Land Use Classification

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)]()
[![Flask](https://img.shields.io/badge/Flask-Web%20Application-black.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

A deep learning-based satellite image classification system that identifies land-use categories from EuroSAT imagery using Optical, SAR, and Multimodal Fusion models built with ResNet-18.

The project includes model training, evaluation, and deployment through a Flask-based web application for real-time predictions.

---

## 📌 Project Overview

Remote sensing imagery plays a critical role in:

- Urban Planning
- Agriculture Monitoring
- Environmental Analysis
- Disaster Management
- Land Cover Mapping

This project leverages Deep Learning and Transfer Learning to classify satellite images into 10 land-use categories using the EuroSAT dataset.

Three separate models are implemented:

1. Optical ResNet18 Model
2. SAR ResNet18 Model
3. Multimodal Fusion ResNet18 Model

The Fusion model combines information from multiple image modalities to improve classification performance.

---

## 🚀 Features

✔ Land-use classification from satellite images

✔ Optical image model

✔ SAR image model

✔ Multimodal fusion model

✔ Interactive Flask web application

✔ Real-time prediction results

✔ Confidence score visualization

✔ REST API support

✔ Transfer learning using ResNet18

---

## 🗂 Dataset

This project uses the EuroSAT Dataset.

### Classes

| Class |
|---------|
| AnnualCrop |
| Forest |
| HerbaceousVegetation |
| Highway |
| Industrial |
| Pasture |
| PermanentCrop |
| Residential |
| River |
| SeaLake |

Dataset Size:

- 27,000+ images
- 10 classes
- RGB Satellite Images

---

## 🏗 Model Architecture

### Optical Model

Input:
RGB Satellite Image

Architecture:

ResNet18
↓
Global Average Pooling
↓
Fully Connected Layer
↓
10 Classes

---

### SAR Model

Input:
SAR / Alternative Spectral Representation

Architecture:

ResNet18
↓
Classification Head
↓
10 Classes

---

### Fusion Model

Input:

Optical Features
+
SAR Features

Architecture:

ResNet18 (Optical)
        ↓
Feature Fusion Layer
        ↓
ResNet18 (SAR)
        ↓
Fully Connected Layer
        ↓
Prediction

---

## 📊 Training Pipeline

1. Data Loading
2. Data Augmentation
3. Train/Validation Split
4. Transfer Learning
5. Fine-Tuning
6. Evaluation
7. Model Export

---

## 📁 Project Structure

```text
EuroSAT-Multimodal-Land-Classification/
│
├── app/
│   ├── app.py
│   ├── requirements.txt
│   ├── models/
│   └── templates/
│
├── notebooks/
│   └── EuroSat_Main_Ai.ipynb
│
├── docs/
│   └── screenshots/
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/EuroSAT-Multimodal-Land-Classification.git

cd EuroSAT-Multimodal-Land-Classification
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r app/requirements.txt
```

---

## ▶ Running the Application

```bash
cd app

python app.py
```

Open Browser:

```text
http://localhost:5000
```

---

## 🔌 REST API

### Check Server Status

```http
GET /api/status
```

Response:

```json
{
  "status": "running"
}
```

---

### Get Classes

```http
GET /api/classes
```

---

### Classify Image

```http
POST /api/classify
```

Form Data:

```text
image : image file
model : optical | sar | fusion
```

Example:

```bash
curl -X POST http://localhost:5000/api/classify \
-F "image=@sample.jpg" \
-F "model=fusion"
```

---

## 📈 Evaluation Metrics

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Fusion models generally outperform single-modality approaches due to richer feature representations.

---

## 📷 Application Screenshots

### Home Page

Add screenshot here

```md
![Home](docs/screenshots/home.png)
```

### Prediction Result

```md
![Prediction](docs/screenshots/result.png)
```

---

## 🛠 Technologies Used

### Machine Learning

- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-Learn

### Web Development

- Flask
- HTML
- CSS
- JavaScript

### Visualization

- Matplotlib
- Seaborn

---

## 🔮 Future Improvements

- Deploy on Hugging Face Spaces
- Docker Support
- Mobile-Friendly Interface
- Grad-CAM Explainability
- Model Quantization
- AWS Deployment

---

## 👨‍💻 Author

**Moksh**

B.Tech (Artificial Intelligence & Machine Learning)

GitHub: https://github.com/yourusername

---

## 📜 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project for educational and research purposes.
