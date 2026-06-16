"""
EuroSAT Multimodal Land Classification — Flask Backend
=======================================================
Connects directly to the trained ResNet-18 models from the project notebook.

Usage:
  1. pip install -r requirements.txt
  2. python app.py
  3. Open http://localhost:5000
"""

import os
import io
import json
import logging
import traceback

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# ─── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10

# Base directory for the application (where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]
CLASS_DESCRIPTIONS = {
    'AnnualCrop':           'Annual agricultural fields',
    'Forest':               'Dense tree cover',
    'HerbaceousVegetation': 'Grasslands and shrubs',
    'Highway':              'Roads and highways',
    'Industrial':           'Factories and warehouses',
    'Pasture':              'Open grazing land',
    'PermanentCrop':        'Orchards and vineyards',
    'Residential':          'Urban housing areas',
    'River':                'Water bodies – rivers',
    'SeaLake':              'Sea and lake water bodies',
}
MODEL_PATHS = {
    'optical': os.path.join(BASE_DIR, 'models', 'resnet18_optical_best.pth'),
    'sar':     os.path.join(BASE_DIR, 'models', 'resnet18_sar_best.pth'),
    'fusion':  os.path.join(BASE_DIR, 'models', 'resnet18_fusion_best.pth'),
}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ─── Model Definitions (same as notebook) ──────────────────────────────────────

def build_resnet18(num_classes=10):
    """ResNet-18 with custom head — Model A & B."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


class FusionResNet18(nn.Module):
    """Dual-encoder fusion model — Model C."""
    def __init__(self, num_classes=10):
        super().__init__()
        resnet_opt    = models.resnet18(weights=None)
        self.enc_opt  = nn.Sequential(*list(resnet_opt.children())[:-1])

        resnet_sar    = models.resnet18(weights=None)
        self.enc_sar  = nn.Sequential(*list(resnet_sar.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_opt, x_sar):
        f_opt = self.enc_opt(x_opt)
        f_sar = self.enc_sar(x_sar)
        fused = torch.cat([f_opt, f_sar], dim=1)
        return self.classifier(fused)


# ─── Model Registry (lazy-load + cache) ────────────────────────────────────────

class ModelRegistry:
    def __init__(self):
        self._cache = {}

    def _load(self, model_type):
        path = MODEL_PATHS[model_type]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Weight file not found: '{path}'. "
                f"Place it in the models/ folder."
            )
        logger.info(f"Loading {model_type} model from {path} ...")

        if model_type == 'fusion':
            model = FusionResNet18(num_classes=NUM_CLASSES)
        else:
            model = build_resnet18(num_classes=NUM_CLASSES)

        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        logger.info(f"  ✅ {model_type} loaded ({sum(p.numel() for p in model.parameters()):,} params)")
        return model

    def get(self, model_type):
        if model_type not in self._cache:
            self._cache[model_type] = self._load(model_type)
        return self._cache[model_type]

    def status(self):
        return {k: os.path.exists(v) for k, v in MODEL_PATHS.items()}


registry = ModelRegistry()


# ─── Inference ─────────────────────────────────────────────────────────────────

def preprocess(pil_img):
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return val_transform(pil_img).unsqueeze(0).to(DEVICE)


def run_inference(model_type, pil_img):
    model  = registry.get(model_type)
    tensor = preprocess(pil_img)

    with torch.no_grad():
        if model_type == 'fusion':
            logits = model(tensor, tensor)   # same image for both branches
        else:
            logits = model(tensor)

        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        'prediction':    pred_class,
        'confidence':    round(confidence, 4),
        'description':   CLASS_DESCRIPTIONS[pred_class],
        'probabilities': {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)},
        'model_type':    model_type,
        'device':        str(DEVICE),
    }


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    return jsonify({
        'status':  'ok',
        'device':  str(DEVICE),
        'models':  registry.status(),
        'classes': CLASS_NAMES,
    })


@app.route('/api/classify', methods=['POST'])
def classify():
    """
    POST /api/classify
    Form fields:
      image — image file (PNG / JPG / TIF ...)
      model — 'optical' | 'sar' | 'fusion'   (default: fusion)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file. Send as form field "image".'}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'Empty filename.'}), 400

    model_type = request.form.get('model', 'fusion').lower()
    if model_type not in MODEL_PATHS:
        return jsonify({'error': f'Unknown model "{model_type}". Use optical | sar | fusion.'}), 400

    try:
        pil_img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({'error': f'Cannot open image: {e}'}), 400

    try:
        result = run_inference(model_type, pil_img)
        logger.info(f"→ {result['prediction']} ({result['confidence']*100:.1f}%) [{model_type}]")
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Inference failed: {e}'}), 500


@app.route('/api/classes')
def api_classes():
    return jsonify({'classes': CLASS_NAMES, 'descriptions': CLASS_DESCRIPTIONS})


# ─── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info(f"🛰️  EuroSAT Backend → http://localhost:5000")
    logger.info(f"   Device : {DEVICE}")
    logger.info(f"   Models : {registry.status()}")
    app.run(host='0.0.0.0', port=5000, debug=True)
