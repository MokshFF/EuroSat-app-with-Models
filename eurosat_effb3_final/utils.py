import os
import io
import copy
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config

# ── 1. Early Stopping ────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.triggered = False

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                if self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        return self.triggered

# ── 2. Cosine Warmup Scheduler ────────────────────────────────────────────────
def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── 3. Grad-CAM Explainable AI ────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Setup hooks
        self.hook_a = self.target_layer.register_forward_hook(self.save_activation)
        # Handle register_full_backward_hook in newer PyTorch, fallback to register_backward_hook
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.hook_g = self.target_layer.register_full_backward_hook(self.save_gradient)
        else:
            self.hook_g = self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, *input_tensors, target_class=None):
        self.model.zero_grad()
        output = self.model(*input_tensors)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        # Mean gradients across spatial dimensions (H, W)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        
        # Interpolate map back to the original image dimensions
        H, W = input_tensors[0].shape[2], input_tensors[0].shape[3]
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        
        # Min-Max Normalization
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        return cam.squeeze().detach().cpu().numpy(), target_class

    def remove_hooks(self):
        self.hook_a.remove()
        self.hook_g.remove()

def generate_gradcam_overlay(pil_img, model, target_layer, transform, device, forward_args=None):
    """
    Computes Grad-CAM activations, overlaps the colormapped heatmap over the original
    satellite image, and returns the result as a base64 encoded string.
    """
    model.eval()
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # Initialize hook manager
    grad_cam = GradCAM(model, target_layer)
    try:
        if forward_args is not None:
            cam, target_class = grad_cam(*forward_args)
        else:
            cam, target_class = grad_cam(input_tensor)
    finally:
        grad_cam.remove_hooks()
        
    # Resize original image to fit tensor dimensions
    H, W = input_tensor.shape[2], input_tensor.shape[3]
    orig_resized = pil_img.resize((W, H))
    
    # Colorize activation values using Matplotlib's Jet colormap
    cmap = plt.get_cmap('jet')
    heatmap = cmap(cam)[:, :, :3]  # shape (H, W, 3) [0.0 - 1.0]
    
    # Normalize original image
    orig_np = np.array(orig_resized) / 255.0
    
    # Alpha blend: 45% original satellite image + 55% activation heatmap
    overlay = 0.45 * orig_np + 0.55 * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    # Convert overlay array back to PIL
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
    
    # Convert PIL to base64 jpeg string
    buffered = io.BytesIO()
    overlay_pil.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_base64, target_class

def convert_to_sar_representation(pil_img):
    """
    Converts or synthesizes a SAR (Synthetic Aperture Radar) backscatter profile
    from an image: highlights surface roughness, structural textures, and adds
    speckle characteristics typical of Sentinel-1 C-band radar.
    """
    # Convert to grayscale luminance
    gray = pil_img.convert('L')
    gray_np = np.array(gray, dtype=np.float32) / 255.0
    
    # Multiplicative speckle noise (Gamma/Rayleigh distributed characteristics of SAR)
    np.random.seed(42)
    speckle = np.random.normal(1.0, 0.08, gray_np.shape)
    sar_sim = np.clip(gray_np * speckle, 0.0, 1.0)
    
    # Map to 3-channel SAR false-color (VV, VH, VV/VH ratio composite)
    ch_vv = sar_sim
    ch_vh = np.clip(sar_sim * 0.75 + 0.05, 0.0, 1.0)
    ch_ratio = np.clip(ch_vv / (ch_vh + 1e-4) * 0.5, 0.0, 1.0)
    
    sar_rgb = np.stack([ch_vv, ch_vh, ch_ratio], axis=-1)
    sar_pil = Image.fromarray((sar_rgb * 255).astype(np.uint8))
    return sar_pil

# ── 4. Atmospheric Weather Haze & Noise Simulation ────────────────────────────
def apply_atmospheric_effects(pil_img, haze_level, blur_level):
    """
    Simulates atmospheric disturbances:
    - Haze Level (0-100): opacity blending with constant white haze overlay.
    - Blur Level (0-100): Gaussian blur filter applied to the image.
    """
    if haze_level > 0:
        # Cap simulated haze at 60% opacity to avoid complete washout
        haze_alpha = (haze_level / 100.0) * 0.6
        white_overlay = Image.new("RGB", pil_img.size, (255, 255, 255))
        pil_img = Image.blend(pil_img, white_overlay, alpha=haze_alpha)
        
    if blur_level > 0:
        # Cap blur radius at 5.0
        blur_radius = (blur_level / 100.0) * 5.0
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
    return pil_img
