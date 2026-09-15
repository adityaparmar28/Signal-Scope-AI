import torch
import torch.nn as nn
import timm
from src.model.srm_filters import SRMFilterLayer

class LegacySRMBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = SRMFilterLayer() # We'll recreate the 3-output layer
        # The checkpoint has 'srm.conv.weight' shape [3, 3, 5, 5]
        self.srm.conv = nn.Conv2d(3, 3, 5, padding=2, bias=False)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, 128)
        
    def forward(self, x):
        x = self.srm(x)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LegacyDualBranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_branch = LegacySRMBranch()
        
        self.spatial_branch = nn.Module()
        self.spatial_branch.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        self.spatial_branch.backbone.reset_classifier(0)
        self.spatial_branch.target_layer = nn.Conv2d(320, 1280, 1, bias=False) # mock
        self.spatial_branch.fc = nn.Linear(1280, 256)
        
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        srm_feat = self.freq_branch(x)
        
        # Spatial
        sp_feat = self.spatial_branch.backbone(x)
        # In timm, efficientnet outputs flattened tensor if reset_classifier(0) is called.
        # But maybe it was not? If sp_feat is (B, 1280)
        if len(sp_feat.shape) == 4:
            sp_feat = sp_feat.view(sp_feat.size(0), -1)
        sp_feat_fc = self.spatial_branch.fc(sp_feat)
        
        fused = torch.cat([sp_feat_fc, srm_feat], dim=1)
        logits_1d = self.classifier(fused)
        
        # Convert 1D logits to 2D fake logits for compatibility with softmax
        prob = torch.sigmoid(logits_1d)
        logits = torch.cat([1.0 - prob, prob], dim=1)
        
        # Gradcam needs this
        return {
            'logits': logits,
            'fused_features': fused,
            'spatial_features': sp_feat
        }
        
    def get_spatial_backbone(self):
        return self.spatial_branch.backbone
