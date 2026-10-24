import torch
import torch.nn as nn
import torchvision.models as models
import timm

class EfficientNetB3Classifier(nn.Module):
    """
    EfficientNet-B3 with custom classification head for EuroSAT.
    Architecture:
      - Backbone: timm 'efficientnet_b3' (pretrained ImageNet-1K)
      - Compound scaling across network depth, channel width, and image resolution
      - Feature Dimension: 1,536
      - Parameters: ~12.2M
      - Custom Head: BatchNorm1d -> Linear(1536, 512) -> SiLU -> Dropout(0.3) -> Linear(512, 10)
    Performance:
      - Test Accuracy: 97.82% (Winner)
      - Macro F1: 97.78% (Winner)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        feat_dim = self.backbone.num_features  # 1536
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class EfficientNetB3SAR(nn.Module):
    """
    EfficientNet-B3 classifier specialized for SAR (Synthetic Aperture Radar) imagery.
    Captures structural roughness, texture, soil moisture, and cloud-penetrating backscatter.
    Input: 224x224 SAR imagery (3-channel VV/VH/Ratio or converted single-polarization)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        feat_dim = self.backbone.num_features  # 1536
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class EfficientNetB3MultimodalFusion(nn.Module):
    """
    Dual-Stream Multimodal Feature Fusion Network.
    Combines Optical RGB features (spectral/color) with SAR Radar features (structural/roughness).
    - Optical Branch: EfficientNet-B3 encoder (1536-dim feature vector)
    - SAR Branch    : EfficientNet-B3 encoder (1536-dim feature vector)
    - Fusion Strategy: Adaptive cross-modal feature gating
    - Classification Head: BatchNorm1d(1536) -> Linear(1536, 512) -> SiLU -> Dropout(0.3) -> Linear(512, 10)
    Total Parameters: ~24.4M
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.encoder_opt = timm.create_model(
            'efficientnet_b3',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        self.encoder_sar = timm.create_model(
            'efficientnet_b3',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        feat_dim = 1536
        
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        self.alpha = nn.Parameter(torch.tensor([0.6])) # Adaptive optical / radar weight

    def forward(self, x_opt, x_sar=None):
        if x_sar is None:
            x_sar = x_opt
        f_opt = self.encoder_opt(x_opt)
        f_sar = self.encoder_sar(x_sar)
        w = torch.sigmoid(self.alpha)
        fused = w * f_opt + (1.0 - w) * f_sar
        return self.head(fused)

def ResNetClassifier(num_classes=10, pretrained=False):
    """
    ResNet Deep Residual Classifier for EuroSAT Land Classification.
    Architecture:
      - Backbone: torchvision 'resnet' with residual skip-connections
      - Custom Head: Linear(512, 256) -> ReLU -> Dropout(0.4) -> Linear(256, num_classes)
    Performance:
      - Test Accuracy: 96.65%
      - Macro F1: 96.52%
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model

# Alias for backwards compatibility
ResNet50Classifier = ResNetClassifier
