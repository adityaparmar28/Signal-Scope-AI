import torch
import torch.nn as nn
import numpy as np

class SRMFilterLayer(nn.Module):
    def __init__(self):
        super(SRMFilterLayer, self).__init__()
        
        # 1. First-order edge filter
        filter1 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
        
        # 2. Second-order edge filter
        filter2 = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32)
        
        # 3. Square 3x3 filter center portion
        filter3 = np.array([[-6, 8, -6], [8, -12, 8], [-6, 8, -6]], dtype=np.float32) / 12.0
        
        filters = np.stack([filter1, filter2, filter3], axis=0) # shape (3, 3, 3)
        filters = np.expand_dims(filters, axis=1) # shape (3, 1, 3, 3)
        
        weights = np.tile(filters, (3, 1, 1, 1)) # (9, 1, 3, 3)
        
        self.conv = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, padding=1, bias=False, groups=3)
        self.conv.weight = nn.Parameter(torch.from_numpy(weights), requires_grad=False)
        
    def forward(self, x):
        return self.conv(x)


class SRMBranch(nn.Module):
    def __init__(self, out_features=256):
        super(SRMBranch, self).__init__()
        self.srm_layer = SRMFilterLayer()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, out_features)
        
    def forward(self, x):
        noise = self.srm_layer(x)
        feat = self.conv_layers(noise)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)
