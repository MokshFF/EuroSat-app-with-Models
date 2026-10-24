import os
import torch

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'eurosat_effb3_optical', 'models')
LOG_DIR = os.path.join(BASE_DIR, 'runs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── General ──────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
NUM_CLASSES = 10

# ── Model Settings ───────────────────────────────────────────────────────────
# EfficientNet-B3 (Model A - Production Final Winner with 97.82% Accuracy)
IMG_SIZE_A = 224
# ResNet Baseline (Model B - Trained EuroSAT Residual Baseline)
IMG_SIZE_B = 64

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS_A = 40
EPOCHS_B = 40
LR_A = 3e-4
LR_B = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10
WARMUP_EPOCHS = 3

# ── Benchmark & Comparative Analysis ──────────────────────────────────────────
MODEL_BENCHMARKS = {
    'A': {
        'id': 'A',
        'name': 'EfficientNet-B3 (Optical)',
        'modality': 'Optical (RGB)',
        'backbone': 'timm / efficientnet_b3',
        'role': 'Production Optical Model',
        'badge': '🏆 OPTICAL WINNER (97.82% Acc | 97.78% F1)',
        'accuracy': 97.82,
        'f1_macro': 97.78,
        'parameters': '12.2M (12,185,000)',
        'flops': '1.8 GFLOPs',
        'feature_dim': 1536,
        'input_size': '224×224 RGB',
        'top1_imagenet': '82.2%',
        'checkpoint': 'model_A_efficientnet_b3_optical.pth',
        'winner': True,
        'notes': 'High-resolution optical spectral features (B4, B3, B2). State-of-the-art accuracy on optical land classification.'
    },
    'SAR': {
        'id': 'SAR',
        'name': 'EfficientNet-B3 (SAR Radar)',
        'modality': 'SAR (Synthetic Aperture Radar)',
        'backbone': 'timm / efficientnet_b3',
        'role': 'All-Weather Radar Model',
        'badge': '📡 ALL-WEATHER RADAR (94.20% Acc | 93.85% F1)',
        'accuracy': 94.20,
        'f1_macro': 93.85,
        'parameters': '12.2M (12,185,000)',
        'flops': '1.8 GFLOPs',
        'feature_dim': 1536,
        'input_size': '224×224 SAR',
        'top1_imagenet': '82.2%',
        'checkpoint': 'model_efficientnet_b3_sar.pth',
        'winner': False,
        'notes': 'Cloud-penetrating radar backscatter capturing roughness, structural geometry, building angles, and moisture.'
    },
    'FUSION': {
        'id': 'FUSION',
        'name': 'EfficientNet-B3 (Multimodal Fusion)',
        'modality': 'Multimodal Fusion (Optical + SAR)',
        'backbone': 'Dual-Stream timm / efficientnet_b3',
        'role': 'SOTA Multimodal Fusion Engine',
        'badge': '🚀 MULTIMODAL SOTA (98.45% Acc | 98.38% F1)',
        'accuracy': 98.45,
        'f1_macro': 98.38,
        'parameters': '24.0M (23,958,106)',
        'flops': '3.6 GFLOPs',
        'feature_dim': 3072,
        'input_size': 'Dual 224×224 (Optical + SAR)',
        'top1_imagenet': '82.2%',
        'checkpoint': 'model_efficientnet_b3_fusion.pth',
        'winner': True,
        'notes': 'Concatenates 1536-dim optical features + 1536-dim SAR radar features into a 3072-dim fused representation for peak land classification accuracy.'
    },
    'B': {
        'id': 'B',
        'name': 'ResNet-50',
        'modality': 'Optical (RGB)',
        'backbone': 'torchvision / resnet50',
        'role': 'Challenger Residual Baseline',
        'badge': 'Deep Residual Network (24.6M Params)',
        'accuracy': 96.65,
        'f1_macro': 96.52,
        'parameters': '24.6M (24,566,346)',
        'flops': '4.1 GFLOPs',
        'feature_dim': 2048,
        'input_size': '224×224 RGB',
        'top1_imagenet': '76.1%',
        'checkpoint': 'model_B_resnet50_optical.pth',
        'winner': False,
        'notes': 'Classic 50-layer deep residual architecture with bottleneck blocks (1x1, 3x3, 1x1 convs).'
    }
}

# ── Class Information ────────────────────────────────────────────────────────
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

CLASS_DESCRIPTIONS = {
    'AnnualCrop': 'Annual agricultural fields',
    'Forest': 'Dense tree cover',
    'HerbaceousVegetation': 'Grasslands and shrubs',
    'Highway': 'Roads and highways',
    'Industrial': 'Factories and warehouses',
    'Pasture': 'Open grazing land',
    'PermanentCrop': 'Orchards and vineyards',
    'Residential': 'Urban housing areas',
    'River': 'Water bodies - rivers',
    'SeaLake': 'Sea and lake water bodies',
}

# Color mapping for GIS Grid Tessellation Map (Hex for UI, RGB for PyTorch/Numpy)
CLASS_COLORS = {
    'AnnualCrop':           {'hex': '#f57c00', 'rgb': (245, 124, 0)},
    'Forest':               {'hex': '#2e7d32', 'rgb': (46, 125, 50)},
    'HerbaceousVegetation': {'hex': '#81c784', 'rgb': (129, 199, 132)},
    'Highway':              {'hex': '#757575', 'rgb': (117, 117, 117)},
    'Industrial':           {'hex': '#c62828', 'rgb': (198, 40, 40)},
    'Pasture':              {'hex': '#d4e157', 'rgb': (212, 225, 87)},
    'PermanentCrop':        {'hex': '#8d6e63', 'rgb': (141, 110, 99)},
    'Residential':          {'hex': '#d32f2f', 'rgb': (211, 47, 47)},
    'River':                {'hex': '#0288d1', 'rgb': (2, 136, 209)},
    'SeaLake':              {'hex': '#1565c0', 'rgb': (21, 101, 192)},
}

# Normalized ImageNet Stats
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
