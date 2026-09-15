"""
SignalScope Dual-Branch Backbone
Combines Spatial Semantic Features with High-Frequency/SRM Residual Features
to maximize generalization across unseen generative models (GANs, Diffusion, Midjourney, Flux).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------
# 1. Spatial Rich Models (SRM) High-Pass Residual Filter Bank
# --------------------------------------------------------------------------
class SRMConv2d(nn.Module):
    """
    SRM high-pass filter bank that extracts noise residuals and upsampling artifacts
    by suppressing low-frequency semantic content.
    """
    def __init__(self):
        super().__init__()
        # Define 3 standard SRM kernels (5x5)
        # 1. 1st order derivative (horizontal)
        srm1 = np.array([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32) / 4.0

        # 2. 2nd order Laplacian
        srm2 = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32) / 12.0

        # 3. Square edge filter
        srm3 = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.float32) / 2.0

        # Shape: (3 filters, 1 channel, 5, 5)
        filters = np.stack([srm1, srm2, srm3], axis=0)[:, np.newaxis, :, :]
        # Replicate for 3 input RGB channels -> (3, 3, 5, 5)
        filters = np.repeat(filters, 3, axis=1) / 3.0

        self.conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv.weight = nn.Parameter(torch.tensor(filters, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        return self.conv(x)


# --------------------------------------------------------------------------
# 2. Frequency / Residual Feature Branch
# --------------------------------------------------------------------------
class FrequencyResidualBranch(nn.Module):
    """
    Processes high-pass filtered SRM residuals to capture periodic spectral
    fingerprints and checkerboard artifacts left by diffusion/GAN decoders.
    """
    def __init__(self, out_features=128):
        super().__init__()
        self.srm = SRMConv2d()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x):
        residuals = self.srm(x)
        features = self.conv_block(residuals)
        features = torch.flatten(features, 1)
        return self.fc(features)


# --------------------------------------------------------------------------
# 3. Spatial Semantic Feature Branch
# --------------------------------------------------------------------------
class SpatialSemanticBranch(nn.Module):
    """
    Captures macro-level visual inconsistencies (lighting, geometry, textures).
    Uses pretrained EfficientNet-B0 backbone for robust real-world generalization.
    """
    def __init__(self, out_features=256):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        in_features = getattr(self.backbone, 'num_features', 1280)
        self.target_layer = self.backbone.conv_head  # Used for Layer-CAM explainability
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        pooled = self.backbone.forward_head(feat, pre_logits=True)
        return self.fc(pooled), feat


# --------------------------------------------------------------------------
# 4. SignalScope Dual-Branch Detector
# --------------------------------------------------------------------------
class SignalScopeDetector(nn.Module):
    """
    Unified dual-branch architecture for real vs. AI-generated image detection.
    Fuses spatial visual cues with high-frequency residual signatures.
    """
    def __init__(self, spatial_dim=256, freq_dim=128):
        super().__init__()
        self.spatial_branch = SpatialSemanticBranch(out_features=spatial_dim)
        self.freq_branch = FrequencyResidualBranch(out_features=freq_dim)
        
        # Multimodal fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, 1)  # Single raw logit: > 0 means AI-generated
        )

    def forward(self, x):
        spatial_feats, feature_map = self.spatial_branch(x)
        freq_feats = self.freq_branch(x)
        fused = torch.cat([spatial_feats, freq_feats], dim=1)
        logit = self.classifier(fused)
        return logit, feature_map

    def predict_proba(self, x, temperature=1.0):
        """Returns calibrated probability of being AI-generated in [0.0, 1.0]."""
        with torch.no_grad():
            logit, _ = self.forward(x)
            prob = torch.sigmoid(logit / temperature)
        return prob
