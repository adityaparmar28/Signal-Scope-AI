import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

def get_target_layer(model):
    if hasattr(model, 'spatial_branch') and hasattr(model.spatial_branch, 'conv_head'):
        return model.spatial_branch.conv_head
    elif hasattr(model, 'spatial_branch') and hasattr(model.spatial_branch, 'backbone') and hasattr(model.spatial_branch.backbone, 'conv_head'):
        return model.spatial_branch.backbone.conv_head
    elif hasattr(model, 'get_spatial_backbone'):
        backbone = model.get_spatial_backbone()
        return list(backbone.children())[-2]
    return list(model.children())[-2]

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict) and 'logits' in out:
            return out['logits']
        return out

class SignalScopeGradCAM:
    def __init__(self, model, device='cpu'):
        self.device = device
        self.wrapped_model = ModelWrapper(model).to(device)
        self.wrapped_model.eval()
        self.target_layer = [get_target_layer(model)]
        self.cam = GradCAM(model=self.wrapped_model, target_layers=self.target_layer)

    def generate(self, image_tensor, target_class=None):
        grayscale_cam = self.cam(input_tensor=image_tensor.to(self.device), targets=target_class)
        return {
            'heatmap': grayscale_cam[0, :],
            'target_class': target_class
        }

    def overlay(self, original_image_np, heatmap):
        return show_cam_on_image(original_image_np, heatmap, use_rgb=True)

    def save_visualization(self, original_image_path, heatmap, output_path):
        img = Image.open(original_image_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = cv2.resize(img_np, (heatmap.shape[1], heatmap.shape[0]))
        overlay_img = self.overlay(img_np, heatmap)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(overlay_img)
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
