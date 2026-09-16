import os
import json
import torch
import torch.nn as nn
import timm
from scipy.optimize import minimize
from src.model.srm_filters import SRMBranch

class DualBranchNet(nn.Module):
    def __init__(self, num_classes=2, attribution_classes=0, backbone='efficientnet_b4', pretrained=True):
        super(DualBranchNet, self).__init__()
        
        # Spatial branch (EfficientNet)
        self.spatial_branch = timm.create_model(backbone, pretrained=pretrained)
        spatial_features = self.spatial_branch.num_features
        self.spatial_branch.reset_classifier(0)
        
        # SRM Branch
        self.srm_branch = SRMBranch(out_features=256)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(spatial_features + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
        # Attribution head
        if attribution_classes > 0:
            self.attribution_head = nn.Linear(256, attribution_classes)
        else:
            self.attribution_head = None

    def forward(self, x):
        spatial_feat = self.spatial_branch(x)
        srm_feat = self.srm_branch(x)
        
        fused = self.fusion(torch.cat([spatial_feat, srm_feat], dim=1))
        logits = self.classifier(fused)
        
        result = {
            'logits': logits,
            'fused_features': fused,
            'spatial_features': spatial_feat
        }
        
        if self.attribution_head is not None:
            result['attribution_logits'] = self.attribution_head(fused)
            
        return result

    def get_spatial_backbone(self):
        return self.spatial_branch


class TemperatureScaler:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def calibrate(self, logits, labels):
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        def eval_loss(temp):
            t = torch.tensor(temp[0], dtype=torch.float32)
            scaled_logits = torch.tensor(logits_np) / t
            loss = nn.CrossEntropyLoss()(scaled_logits, torch.tensor(labels_np))
            return loss.item()
            
        res = minimize(eval_loss, [1.0], bounds=[(0.1, 10.0)], method='L-BFGS-B')
        self.temperature = float(res.x[0])
        return self.temperature

    def scale(self, logits):
        return logits / self.temperature

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'temperature': self.temperature}, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                self.temperature = data.get('temperature', 1.0)


def load_model(checkpoint_path, device='cpu', num_classes=2, attribution_classes=0):
    from src.model.legacy_model import LegacyDualBranchNet
    model = LegacyDualBranchNet()
    scaler = TemperatureScaler()
    
    if os.path.isdir(checkpoint_path):
        weights_path = os.path.join(checkpoint_path, 'model.pt')
        temp_path = os.path.join(checkpoint_path, 'temperature.json')
    else:
        weights_path = checkpoint_path
        temp_path = os.path.join(os.path.dirname(checkpoint_path), 'temperature.json')
        
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        
    scaler.load(temp_path)
    model.to(device)
    
    return model, scaler
