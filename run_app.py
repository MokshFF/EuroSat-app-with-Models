"""
EuroSAT Satellite Land Intelligence & Architecture Suite
=========================================================
Root Launcher for the Enhanced Web Application:
- Compares EfficientNet-B3 (WINNER: 97.82% Acc | 97.78% F1) vs ResNet-50 (Challenger: 96.65% Acc)
- Features Single Patch Analysis, Grad-CAM, Weather Robustness Sim, GIS Grid Tessellation, and Resume Dashboard.

Usage:
    python run_app.py
"""

import os
import sys

# Configure stdout for UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Set directory to web app folder
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'eurosat_effb3_final', 'eurosat_effb3_optical')
sys.path.insert(0, os.path.join(current_dir, 'eurosat_effb3_final'))
sys.path.insert(0, app_dir)

os.chdir(app_dir)

if __name__ == '__main__':
    from app import app
    print("=" * 78)
    print(" [EuroSAT] Multimodal Earth Observation Suite: Optical & SAR Radar")
    print("=" * 78)
    print(" [SOTA FUSION]  : EfficientNet-B3 Multimodal Fusion (98.45% Acc | 98.38% F1)")
    print(" [OPTICAL SOTA] : EfficientNet-B3 Sentinel-2 RGB    (97.82% Acc | 97.78% F1)")
    print(" [SAR RADAR]    : EfficientNet-B3 Sentinel-1 Radar  (94.20% Acc | 93.85% F1)")
    print(" [CHALLENGER]   : ResNet-50 Optical Baseline       (96.65% Acc | 96.52% F1)")
    print(" [WEB PORTAL]   : http://localhost:5000")
    print("=" * 78)
    app.run(host='0.0.0.0', port=5000, debug=False)
