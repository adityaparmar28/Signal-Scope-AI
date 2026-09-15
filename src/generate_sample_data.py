"""
Helper script to generate minimal synthetic test samples for format and reproducibility checks.
"""

import os
import numpy as np
from PIL import Image, ImageDraw


def generate_sample_dataset(base_dir="data/sample_val"):
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # 1. Create 5 sample "real" photos (natural gradient + realistic high-frequency noise)
    for i in range(1, 6):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        # Smooth natural sky/ground gradient
        for y in range(256):
            arr[y, :, 0] = int(100 + (y / 256.0) * 120)
            arr[y, :, 1] = int(120 + (y / 256.0) * 100)
            arr[y, :, 2] = int(150 + (y / 256.0) * 80)
        # Add natural camera sensor grain
        noise = np.random.normal(0, 8, (256, 256, 3)).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.ellipse([40 + i*10, 50, 100 + i*10, 110], fill=(220, 180, 140))
        img.save(os.path.join(real_dir, f"sample_real_{i}.jpg"), quality=95)

    # 2. Create 5 sample "synthetic" images (with deconvolution/checkerboard artifacts & warped shapes)
    for i in range(1, 6):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        # Add high-frequency periodic grid / checkerboard pattern typical of GAN/diffusion upscalers
        for y in range(256):
            for x in range(256):
                val = int(128 + 40 * np.sin(x / 4.0) * np.cos(y / 4.0))
                arr[y, x, :] = np.clip(val, 0, 255)

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        # Warped unnatural geometric artifact
        draw.polygon([(30, 30), (180, 60), (140, 200), (50, 170)], fill=(255, 100, 180))
        draw.text((60, 100), "AI Artefact Sample", fill=(255, 255, 255))
        img.save(os.path.join(fake_dir, f"sample_synthetic_{i}.jpg"), quality=95)

    print(f"[OK] Generated sample dataset in '{base_dir}': 5 real and 5 synthetic images.")


if __name__ == "__main__":
    generate_sample_dataset()
