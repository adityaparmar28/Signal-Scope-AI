"""
SignalScope Dataset & Augmentation Pipeline
Includes aggressive degradation augmentations (JPEG compression, blur, noise)
to ensure robustness and generalize to unseen generators (Bonus C & Section 4.1).
"""

import io
import os
import random
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class JPEGCompressionAugmentation:
    """Simulates real-world social media and messaging compression."""
    def __init__(self, quality_range=(30, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            quality = random.randint(*self.quality_range)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")
        return img


class GaussianBlurAugmentation:
    """Simulates optical and motion blurring."""
    def __init__(self, radius_range=(0.5, 2.0), p=0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img


def get_training_transforms():
    return transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        JPEGCompressionAugmentation(quality_range=(35, 90), p=0.5),
        GaussianBlurAugmentation(radius_range=(0.5, 1.8), p=0.3),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class RealVsAIDataset(Dataset):
    """
    Standard PyTorch dataset for Real vs Synthetic image classification.
    Supports directory-based loading:
        root_dir/
            real/
            fake/
    or custom label mappings.
    """
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        if os.path.exists(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    self.samples.append((os.path.join(real_dir, fname), 0))  # 0 = Real

        if os.path.exists(fake_dir):
            for fname in os.listdir(fake_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    self.samples.append((os.path.join(fake_dir, fname), 1))  # 1 = Fake/AI

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32), img_path
