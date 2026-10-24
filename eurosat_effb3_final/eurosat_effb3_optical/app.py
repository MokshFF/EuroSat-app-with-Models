import os
import io
import sys
import time
import logging
import traceback
import base64
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Configure stdout for UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Append parent directory to sys.path to import config, models, utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import models
import utils

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}

# ── Inference Transforms ──────────────────────────────────────────────────────
val_transform_A = transforms.Compose([
    transforms.Resize((config.IMG_SIZE_A, config.IMG_SIZE_A)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
])

val_transform_B = transforms.Compose([
    transforms.Resize((config.IMG_SIZE_B, config.IMG_SIZE_B)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
])

# ── Global Model Registry ─────────────────────────────────────────────────────
model_A = None       # EfficientNet-B3 (Optical RGB)
model_sar = None     # EfficientNet-B3 (SAR Radar)
model_fusion = None  # EfficientNet-B3 (Multimodal Optical + SAR Fusion)
model_B = None       # ResNet-50 (Optical Challenger)

MODEL_A_READY = False
MODEL_SAR_READY = False
MODEL_FUSION_READY = False
MODEL_B_READY = False

def load_models():
    global model_A, model_sar, model_fusion, model_B
    global MODEL_A_READY, MODEL_SAR_READY, MODEL_FUSION_READY, MODEL_B_READY
    
    # 1. Model A: EfficientNet-B3 Optical
    path_A = os.path.join(config.MODEL_DIR, 'model_A_efficientnet_b3_optical.pth')
    if os.path.exists(path_A):
        try:
            logger.info(f"Loading Model A (EfficientNet-B3 Optical) from {path_A}...")
            model_A = models.EfficientNetB3Classifier(num_classes=config.NUM_CLASSES)
            state = torch.load(path_A, map_location=config.DEVICE)
            model_A.load_state_dict(state)
            model_A.to(config.DEVICE).eval()
            MODEL_A_READY = True
            logger.info("  [OK] Model A (Optical) ready.")
        except Exception as e:
            logger.error(f"  [ERR] Failed to load Model A weights: {e}")
    else:
        logger.warning(f"  [WARN] Model A weights not found at {path_A}")

    # 2. Model SAR: EfficientNet-B3 SAR Radar
    path_sar = os.path.join(config.MODEL_DIR, 'model_efficientnet_b3_sar.pth')
    if os.path.exists(path_sar):
        try:
            logger.info(f"Loading Model SAR (EfficientNet-B3 Radar) from {path_sar}...")
            model_sar = models.EfficientNetB3SAR(num_classes=config.NUM_CLASSES)
            state = torch.load(path_sar, map_location=config.DEVICE)
            model_sar.load_state_dict(state)
            model_sar.to(config.DEVICE).eval()
            MODEL_SAR_READY = True
            logger.info("  [OK] Model SAR (Radar) ready.")
        except Exception as e:
            logger.error(f"  [ERR] Failed to load Model SAR weights: {e}")
    else:
        logger.warning(f"  [WARN] Model SAR weights not found at {path_sar}")

    # 3. Model Fusion: EfficientNet-B3 Multimodal Fusion (Optical + SAR)
    path_fusion = os.path.join(config.MODEL_DIR, 'model_efficientnet_b3_fusion.pth')
    if os.path.exists(path_fusion):
        try:
            logger.info(f"Loading Model Fusion (EfficientNet-B3 Optical+SAR) from {path_fusion}...")
            model_fusion = models.EfficientNetB3MultimodalFusion(num_classes=config.NUM_CLASSES)
            state = torch.load(path_fusion, map_location=config.DEVICE)
            model_fusion.load_state_dict(state)
            model_fusion.to(config.DEVICE).eval()
            MODEL_FUSION_READY = True
            logger.info("  [OK] Model Fusion (Optical + SAR) ready.")
        except Exception as e:
            logger.error(f"  [ERR] Failed to load Model Fusion weights: {e}")
    else:
        logger.warning(f"  [WARN] Model Fusion weights not found at {path_fusion}")

    # 4. Model B: ResNet-50 Optical Baseline
    path_B = os.path.join(config.MODEL_DIR, 'model_B_resnet50_optical.pth')
    if os.path.exists(path_B):
        try:
            logger.info(f"Loading Model B (ResNet-50) from {path_B}...")
            model_B = models.ResNet50Classifier(num_classes=config.NUM_CLASSES, pretrained=False)
            state = torch.load(path_B, map_location=config.DEVICE)
            model_B.load_state_dict(state)
            model_B.to(config.DEVICE).eval()
            MODEL_B_READY = True
            logger.info("  [OK] Model B (ResNet-50) ready.")
        except Exception as e:
            logger.error(f"  [ERR] Failed to load Model B weights: {e}")
    else:
        logger.warning(f"  [WARN] Model B weights not found at {path_B}")

load_models()

def allowed_file(filename):
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'ok',
        'device': str(config.DEVICE),
        'model_A_ready': MODEL_A_READY,
        'model_sar_ready': MODEL_SAR_READY,
        'model_fusion_ready': MODEL_FUSION_READY,
        'model_B_ready': MODEL_B_READY,
        'final_model': 'EfficientNet-B3 Multimodal (Optical + SAR) & Optical SOTA',
        'final_accuracy': config.MODEL_BENCHMARKS['FUSION']['accuracy'],
        'final_f1': config.MODEL_BENCHMARKS['FUSION']['f1_macro'],
        'benchmarks': config.MODEL_BENCHMARKS,
        'classes': config.CLASS_NAMES
    })

@app.route('/api/benchmarks')
def api_benchmarks():
    return jsonify({
        'status': 'ok',
        'models': config.MODEL_BENCHMARKS,
        'summary': {
            'winner': 'EfficientNet-B3 (Multimodal Fusion & Optical)',
            'fusion_accuracy': config.MODEL_BENCHMARKS['FUSION']['accuracy'],
            'optical_accuracy': config.MODEL_BENCHMARKS['A']['accuracy'],
            'sar_accuracy': config.MODEL_BENCHMARKS['SAR']['accuracy'],
            'resnet50_accuracy': config.MODEL_BENCHMARKS['B']['accuracy'],
            'accuracy_delta_optical_vs_resnet': round(config.MODEL_BENCHMARKS['A']['accuracy'] - config.MODEL_BENCHMARKS['B']['accuracy'], 2),
            'fusion_gain': round(config.MODEL_BENCHMARKS['FUSION']['accuracy'] - config.MODEL_BENCHMARKS['A']['accuracy'], 2),
            'param_savings_effb3_vs_resnet': 50.4,
            'reason': 'Compound scaling allocates balanced capacity across depth, width, and resolution. Dual-stream fusion synergizes optical spectral reflectance with SAR radar structural backscatter for peak reliability under any cloud or lighting conditions.'
        }
    })

@app.route('/api/classify', methods=['POST'])
def classify():
    """
    Classifies satellite imagery using the selected modality/model:
    - Optical RGB (EfficientNet-B3)
    - SAR Radar Backscatter (EfficientNet-B3)
    - Multimodal Fusion (Optical + SAR Dual-Stream EfficientNet-B3)
    - ResNet-50 (Baseline Optical)
    """
    modality = request.form.get('modality', '').lower().strip()
    model_type = request.form.get('model_type', '').upper().strip()

    # Map aliases
    if not modality:
        if model_type == 'SAR':
            modality = 'sar'
        elif model_type == 'FUSION':
            modality = 'fusion'
        elif model_type == 'B':
            modality = 'resnet50'
        else:
            modality = 'optical'
    elif modality in ('a', 'optical_rgb', 'optical-rgb'):
        modality = 'optical'
    elif modality in ('b', 'resnet', 'resnet_50'):
        modality = 'resnet50'

    # Check input files
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent in multipart field "image".'}), 400
        
    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'Empty filename for primary image.'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format. Please use PNG, JPG, or TIF.'}), 400

    try:
        pil_img = Image.open(io.BytesIO(file.read()))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
    except Exception as e:
        return jsonify({'error': f"Failed to parse primary image: {e}"}), 400

    # Optional SAR file for Multimodal Fusion mode
    pil_img_sar = None
    if 'image_sar' in request.files and request.files['image_sar'].filename != '':
        sar_file = request.files['image_sar']
        if allowed_file(sar_file.filename):
            try:
                pil_img_sar = Image.open(io.BytesIO(sar_file.read()))
                if pil_img_sar.mode != 'RGB':
                    pil_img_sar = pil_img_sar.convert('RGB')
            except Exception as e:
                logger.warning(f"Could not parse uploaded SAR file: {e}. Falling back to synthesized radar.")

    try:
        start_time = time.perf_counter()

        # ── 1. Multimodal Fusion Mode ─────────────────────────────────────────
        if modality == 'fusion':
            if not MODEL_FUSION_READY or model_fusion is None:
                return jsonify({'error': 'EfficientNet-B3 Multimodal Fusion model is not ready.'}), 503
            
            # If explicit SAR patch not provided, synthesize SAR backscatter
            if pil_img_sar is None:
                pil_img_sar = utils.convert_to_sar_representation(pil_img)
            
            tensor_opt = val_transform_A(pil_img).unsqueeze(0).to(config.DEVICE)
            tensor_sar = val_transform_A(pil_img_sar).unsqueeze(0).to(config.DEVICE)

            with torch.no_grad():
                logits = model_fusion(tensor_opt, tensor_sar)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            pred_idx = int(np.argmax(probs))
            pred_class = config.CLASS_NAMES[pred_idx]

            # Dual Grad-CAM overlays: Optical attention + SAR attention
            cam_opt_b64 = None
            cam_sar_b64 = None
            try:
                cam_opt_b64, _ = utils.generate_gradcam_overlay(
                    pil_img, model_fusion, model_fusion.encoder_opt.conv_head,
                    val_transform_A, config.DEVICE, forward_args=(tensor_opt, tensor_sar)
                )
            except Exception as e:
                logger.warning(f"Grad-CAM Optical branch error: {e}")

            try:
                cam_sar_b64, _ = utils.generate_gradcam_overlay(
                    pil_img_sar, model_fusion, model_fusion.encoder_sar.conv_head,
                    val_transform_A, config.DEVICE, forward_args=(tensor_opt, tensor_sar)
                )
            except Exception as e:
                logger.warning(f"Grad-CAM SAR branch error: {e}")

            # Encode processed SAR image
            buf_sar = io.BytesIO()
            pil_img_sar.save(buf_sar, format="JPEG")
            sar_b64 = base64.b64encode(buf_sar.getvalue()).decode('utf-8')

            buf_opt = io.BytesIO()
            pil_img.save(buf_opt, format="JPEG")
            opt_b64 = base64.b64encode(buf_opt.getvalue()).decode('utf-8')

            return jsonify({
                'modality': 'fusion',
                'model': 'EfficientNet-B3 Multimodal Fusion (Optical + SAR)',
                'badge': config.MODEL_BENCHMARKS['FUSION']['badge'],
                'prediction': pred_class,
                'confidence': round(float(probs[pred_idx]), 4),
                'description': config.CLASS_DESCRIPTIONS[pred_class],
                'probabilities': {config.CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(config.NUM_CLASSES)},
                'latency_ms': round(latency_ms, 2),
                'processed_image': opt_b64,
                'processed_image_sar': sar_b64,
                'gradcam': cam_opt_b64,
                'gradcam_sar': cam_sar_b64,
                'benchmark': config.MODEL_BENCHMARKS['FUSION'],
                'device': str(config.DEVICE)
            })

        # ── 2. SAR Radar Mode ────────────────────────────────────────────────
        elif modality == 'sar':
            if not MODEL_SAR_READY or model_sar is None:
                return jsonify({'error': 'EfficientNet-B3 SAR model is not ready.'}), 503

            tensor_sar = val_transform_A(pil_img).unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                logits = model_sar(tensor_sar)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            pred_idx = int(np.argmax(probs))
            pred_class = config.CLASS_NAMES[pred_idx]

            gradcam_b64 = None
            try:
                gradcam_b64, _ = utils.generate_gradcam_overlay(
                    pil_img, model_sar, model_sar.backbone.conv_head,
                    val_transform_A, config.DEVICE
                )
            except Exception as e:
                logger.warning(f"GradCAM SAR error: {e}")

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return jsonify({
                'modality': 'sar',
                'model': 'EfficientNet-B3 (SAR Radar)',
                'badge': config.MODEL_BENCHMARKS['SAR']['badge'],
                'prediction': pred_class,
                'confidence': round(float(probs[pred_idx]), 4),
                'description': config.CLASS_DESCRIPTIONS[pred_class],
                'probabilities': {config.CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(config.NUM_CLASSES)},
                'latency_ms': round(latency_ms, 2),
                'processed_image': img_b64,
                'gradcam': gradcam_b64,
                'benchmark': config.MODEL_BENCHMARKS['SAR'],
                'device': str(config.DEVICE)
            })

        # ── 3. ResNet-50 Challenger Mode ─────────────────────────────────────
        elif modality == 'resnet50':
            if not MODEL_B_READY or model_B is None:
                return jsonify({'error': 'ResNet-50 model is not ready.'}), 503

            tensor_res = val_transform_B(pil_img).unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                logits = model_B(tensor_res)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            pred_idx = int(np.argmax(probs))
            pred_class = config.CLASS_NAMES[pred_idx]

            gradcam_b64 = None
            try:
                cam_layer = model_B.layer4[-1]
                gradcam_b64, _ = utils.generate_gradcam_overlay(
                    pil_img, model_B, cam_layer,
                    val_transform_B, config.DEVICE
                )
            except Exception as e:
                logger.warning(f"GradCAM ResNet error: {e}")

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return jsonify({
                'modality': 'resnet50',
                'model': 'ResNet-50 (Challenger Baseline)',
                'badge': config.MODEL_BENCHMARKS['B']['badge'],
                'prediction': pred_class,
                'confidence': round(float(probs[pred_idx]), 4),
                'description': config.CLASS_DESCRIPTIONS[pred_class],
                'probabilities': {config.CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(config.NUM_CLASSES)},
                'latency_ms': round(latency_ms, 2),
                'processed_image': img_b64,
                'gradcam': gradcam_b64,
                'benchmark': config.MODEL_BENCHMARKS['B'],
                'device': str(config.DEVICE)
            })

        # ── 4. Optical Mode (Default Hero: EfficientNet-B3 Optical) ───────────
        else:
            if not MODEL_A_READY or model_A is None:
                return jsonify({'error': 'EfficientNet-B3 Optical model is not ready.'}), 503

            tensor_opt = val_transform_A(pil_img).unsqueeze(0).to(config.DEVICE)
            with torch.no_grad():
                logits = model_A(tensor_opt)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            pred_idx = int(np.argmax(probs))
            pred_class = config.CLASS_NAMES[pred_idx]

            gradcam_b64 = None
            try:
                gradcam_b64, _ = utils.generate_gradcam_overlay(
                    pil_img, model_A, model_A.backbone.conv_head,
                    val_transform_A, config.DEVICE
                )
            except Exception as e:
                logger.warning(f"GradCAM EfficientNet-B3 Optical error: {e}")

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return jsonify({
                'modality': 'optical',
                'model': 'EfficientNet-B3 (Optical RGB - SOTA Winner)',
                'badge': config.MODEL_BENCHMARKS['A']['badge'],
                'prediction': pred_class,
                'confidence': round(float(probs[pred_idx]), 4),
                'description': config.CLASS_DESCRIPTIONS[pred_class],
                'probabilities': {config.CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(config.NUM_CLASSES)},
                'latency_ms': round(latency_ms, 2),
                'processed_image': img_b64,
                'gradcam': gradcam_b64,
                'benchmark': config.MODEL_BENCHMARKS['A'],
                'device': str(config.DEVICE)
            })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'error': f"Inference pipeline failed: {e}"}), 500

@app.route('/api/compare', methods=['POST'])
def compare_models():
    """
    Comprehensive multi-sensor and multi-model benchmark:
    Executes EfficientNet-B3 Optical, EfficientNet-B3 SAR Radar,
    EfficientNet-B3 Multimodal Fusion, and ResNet-50 simultaneously.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent in multipart field "image".'}), 400
        
    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file format.'}), 400

    try:
        pil_img = Image.open(io.BytesIO(file.read()))
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
    except Exception as e:
        return jsonify({'error': f"Failed to parse image: {e}"}), 400

    # Optional SAR image
    pil_img_sar = None
    if 'image_sar' in request.files and request.files['image_sar'].filename != '':
        try:
            pil_img_sar = Image.open(io.BytesIO(request.files['image_sar'].read()))
            if pil_img_sar.mode != 'RGB':
                pil_img_sar = pil_img_sar.convert('RGB')
        except Exception:
            pass

    if pil_img_sar is None:
        pil_img_sar = utils.convert_to_sar_representation(pil_img)

    buf_opt = io.BytesIO()
    pil_img.save(buf_opt, format="JPEG")
    img_opt_b64 = base64.b64encode(buf_opt.getvalue()).decode('utf-8')

    buf_sar = io.BytesIO()
    pil_img_sar.save(buf_sar, format="JPEG")
    img_sar_b64 = base64.b64encode(buf_sar.getvalue()).decode('utf-8')

    results = {
        'processed_image': img_opt_b64,
        'processed_image_sar': img_sar_b64,
        'model_A': None,        # EfficientNet-B3 Optical
        'model_sar': None,      # EfficientNet-B3 SAR
        'model_fusion': None,   # EfficientNet-B3 Multimodal Fusion
        'model_B': None,        # ResNet-50
        'comparison': None
    }

    tensor_opt = val_transform_A(pil_img).unsqueeze(0).to(config.DEVICE)
    tensor_sar = val_transform_A(pil_img_sar).unsqueeze(0).to(config.DEVICE)
    tensor_res = val_transform_B(pil_img).unsqueeze(0).to(config.DEVICE)

    # 1. EfficientNet-B3 Optical
    if MODEL_A_READY and model_A is not None:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits_A = model_A(tensor_opt)
            probs_A = torch.softmax(logits_A, dim=1).squeeze(0).cpu().numpy()
        lat_A = (time.perf_counter() - t0) * 1000.0
        idx_A = int(np.argmax(probs_A))
        class_A = config.CLASS_NAMES[idx_A]
        cam_A, _ = utils.generate_gradcam_overlay(pil_img, model_A, model_A.backbone.conv_head, val_transform_A, config.DEVICE)
        results['model_A'] = {
            'name': 'EfficientNet-B3 (Optical RGB)',
            'modality': 'Optical (RGB)',
            'prediction': class_A,
            'confidence': round(float(probs_A[idx_A]), 4),
            'description': config.CLASS_DESCRIPTIONS[class_A],
            'probabilities': {config.CLASS_NAMES[i]: round(float(probs_A[i]), 4) for i in range(config.NUM_CLASSES)},
            'latency_ms': round(lat_A, 2),
            'gradcam': cam_A,
            'benchmark': config.MODEL_BENCHMARKS['A']
        }

    # 2. EfficientNet-B3 SAR Radar
    if MODEL_SAR_READY and model_sar is not None:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits_sar = model_sar(tensor_sar)
            probs_sar = torch.softmax(logits_sar, dim=1).squeeze(0).cpu().numpy()
        lat_sar = (time.perf_counter() - t0) * 1000.0
        idx_sar = int(np.argmax(probs_sar))
        class_sar = config.CLASS_NAMES[idx_sar]
        cam_sar, _ = utils.generate_gradcam_overlay(pil_img_sar, model_sar, model_sar.backbone.conv_head, val_transform_A, config.DEVICE)
        results['model_sar'] = {
            'name': 'EfficientNet-B3 (SAR Radar)',
            'modality': 'SAR (Synthetic Aperture Radar)',
            'prediction': class_sar,
            'confidence': round(float(probs_sar[idx_sar]), 4),
            'description': config.CLASS_DESCRIPTIONS[class_sar],
            'probabilities': {config.CLASS_NAMES[i]: round(float(probs_sar[i]), 4) for i in range(config.NUM_CLASSES)},
            'latency_ms': round(lat_sar, 2),
            'gradcam': cam_sar,
            'benchmark': config.MODEL_BENCHMARKS['SAR']
        }

    # 3. EfficientNet-B3 Multimodal Fusion
    if MODEL_FUSION_READY and model_fusion is not None:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits_f = model_fusion(tensor_opt, tensor_sar)
            probs_f = torch.softmax(logits_f, dim=1).squeeze(0).cpu().numpy()
        lat_f = (time.perf_counter() - t0) * 1000.0
        idx_f = int(np.argmax(probs_f))
        class_f = config.CLASS_NAMES[idx_f]
        cam_f_opt, _ = utils.generate_gradcam_overlay(pil_img, model_fusion, model_fusion.encoder_opt.conv_head, val_transform_A, config.DEVICE, forward_args=(tensor_opt, tensor_sar))
        cam_f_sar, _ = utils.generate_gradcam_overlay(pil_img_sar, model_fusion, model_fusion.encoder_sar.conv_head, val_transform_A, config.DEVICE, forward_args=(tensor_opt, tensor_sar))
        results['model_fusion'] = {
            'name': 'EfficientNet-B3 (Multimodal Fusion)',
            'modality': 'Multimodal Fusion (Optical + SAR)',
            'prediction': class_f,
            'confidence': round(float(probs_f[idx_f]), 4),
            'description': config.CLASS_DESCRIPTIONS[class_f],
            'probabilities': {config.CLASS_NAMES[i]: round(float(probs_f[i]), 4) for i in range(config.NUM_CLASSES)},
            'latency_ms': round(lat_f, 2),
            'gradcam': cam_f_opt,
            'gradcam_sar': cam_f_sar,
            'benchmark': config.MODEL_BENCHMARKS['FUSION']
        }

    # 4. ResNet-50 Optical Challenger
    if MODEL_B_READY and model_B is not None:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits_B = model_B(tensor_res)
            probs_B = torch.softmax(logits_B, dim=1).squeeze(0).cpu().numpy()
        lat_B = (time.perf_counter() - t0) * 1000.0
        idx_B = int(np.argmax(probs_B))
        class_B = config.CLASS_NAMES[idx_B]
        cam_B = None
        try:
            cam_layer = model_B.layer4[-1]
            cam_B, _ = utils.generate_gradcam_overlay(pil_img, model_B, cam_layer, val_transform_B, config.DEVICE)
        except Exception as e:
            logger.warning(f"GradCAM ResNet compare error: {e}")
        results['model_B'] = {
            'name': 'ResNet-50 (Optical Challenger)',
            'modality': 'Optical (RGB)',
            'prediction': class_B,
            'confidence': round(float(probs_B[idx_B]), 4),
            'description': config.CLASS_DESCRIPTIONS[class_B],
            'probabilities': {config.CLASS_NAMES[i]: round(float(probs_B[i]), 4) for i in range(config.NUM_CLASSES)},
            'latency_ms': round(lat_B, 2),
            'gradcam': cam_B,
            'benchmark': config.MODEL_BENCHMARKS['B']
        }

    # Consolidated comparison metrics
    if results['model_A'] and results['model_B']:
        agreement = (results['model_A']['prediction'] == results['model_B']['prediction'])
        conf_diff = round(abs(results['model_A']['confidence'] - results['model_B']['confidence']), 4)
        results['comparison'] = {
            'agreement': agreement,
            'agreed_class': results['model_A']['prediction'] if agreement else None,
            'confidence_delta': conf_diff,
            'production_winner': 'EfficientNet-B3 (Multimodal SOTA: 98.45% | Optical: 97.82%)',
            'optical_vs_resnet_gain': round(config.MODEL_BENCHMARKS['A']['accuracy'] - config.MODEL_BENCHMARKS['B']['accuracy'], 2),
            'fusion_vs_resnet_gain': round(config.MODEL_BENCHMARKS['FUSION']['accuracy'] - config.MODEL_BENCHMARKS['B']['accuracy'], 2),
            'param_savings_effb3_vs_resnet': 50.4,
            'flop_savings_effb3_vs_resnet': 56.1
        }

    return jsonify(results)

@app.route('/api/classes')
def api_classes():
    return jsonify({
        'classes': config.CLASS_NAMES, 
        'descriptions': config.CLASS_DESCRIPTIONS,
        'colors': {name: info['hex'] for name, info in config.CLASS_COLORS.items()}
    })

if __name__ == '__main__':
    logger.info("Initializing EuroSAT Intelligence Backend...")
    logger.info(f"Flask API server running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
