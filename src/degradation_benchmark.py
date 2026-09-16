"""
SignalScope Bonus Module C: Robustness to Degradation Benchmark
Applies a systematic battery of real-world degradations (JPEG, blur, downsampling, screenshotting)
and computes a degradation-vs-accuracy analysis (Section 3.2 Bonus C).
"""

import argparse
import io
import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.backbone import SignalScopeDetector


def apply_degradation(img: Image.Image, degradation_type: str, severity: float = 1.0) -> Image.Image:
    """Applies a realistic image degradation."""
    if degradation_type == "pristine":
        return img.copy()

    elif degradation_type == "jpeg":
        # severity in [1.0 (light), 2.0 (medium), 3.0 (heavy)]
        quality = int(max(15, 90 - (severity * 25)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    elif degradation_type == "blur":
        radius = 0.5 * severity
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    elif degradation_type == "downscale":
        # Downscale then upscale back (loss of high-frequency texture)
        w, h = img.size
        factor = 1.0 + (severity * 0.75)
        small = img.resize((int(w / factor), int(h / factor)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    elif degradation_type == "screenshot":
        # Slight crop + re-compression
        w, h = img.size
        crop_px = int(4 * severity)
        cropped = img.crop((crop_px, crop_px, w - crop_px, h - crop_px)).resize((w, h), Image.BICUBIC)
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return img.copy()


def benchmark_degradations(model, image_paths, labels, device):
    """
    Runs the full degradation matrix across all provided samples.
    """
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_battery = [
        ("Pristine / Uncompressed", "pristine", 1.0),
        ("JPEG Compression (Q=65)", "jpeg", 1.0),
        ("JPEG Compression (Q=40)", "jpeg", 2.0),
        ("Heavy JPEG (Q=15)", "jpeg", 3.0),
        ("Downscale + Upscale (2x)", "downscale", 1.5),
        ("Gaussian Blur (r=1.0)", "blur", 2.0),
        ("Re-compressed Screenshot", "screenshot", 1.5),
    ]

    results = []

    for name, deg_type, severity in test_battery:
        correct = 0
        conf_deltas = []
        flips = 0

        for img_path, target in zip(image_paths, labels):
            orig_pil = Image.open(img_path).convert("RGB")
            
            # Baseline inference
            t_orig = transform(orig_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                l_orig, _ = model(t_orig)
                p_orig = torch.sigmoid(l_orig).item()

            # Degraded inference
            deg_pil = apply_degradation(orig_pil, deg_type, severity)
            t_deg = transform(deg_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                l_deg, _ = model(t_deg)
                p_deg = torch.sigmoid(l_deg).item()

            pred = 1 if p_deg >= 0.5 else 0
            if pred == target:
                correct += 1

            conf_deltas.append(abs(p_deg - p_orig))
            if (p_orig >= 0.5) != (p_deg >= 0.5):
                flips += 1

        acc = (correct / len(labels)) * 100 if len(labels) > 0 else 0.0
        avg_delta = float(np.mean(conf_deltas)) if conf_deltas else 0.0
        flip_pct = (flips / len(labels)) * 100 if len(labels) > 0 else 0.0

        results.append({
            "Condition": name,
            "Accuracy": f"{acc:.1f}%",
            "Avg dConfidence": f"{avg_delta:.3f}",
            "Verdict Flip Rate": f"{flip_pct:.1f}%"
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="SignalScope Degradation Robustness Benchmark")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing real/ and fake/ subdirectories")
    parser.add_argument("--weights", type=str, default=None, help="Model checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignalScopeDetector().to(device)
    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))

    # Collect sample paths
    samples = []
    real_dir = os.path.join(args.test_dir, "real")
    fake_dir = os.path.join(args.test_dir, "fake")

    if os.path.exists(real_dir):
        for f in os.listdir(real_dir)[:25]:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                samples.append((os.path.join(real_dir, f), 0))
    if os.path.exists(fake_dir):
        for f in os.listdir(fake_dir)[:25]:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                samples.append((os.path.join(fake_dir, f), 1))

    if not samples:
        print("[!] No sample images found in test_dir.")
        return

    paths, labels = zip(*samples)
    print(f"[*] Running robustness benchmark on {len(samples)} images...")
    benchmarks = benchmark_degradations(model, paths, labels, device)

    print("\n" + "=" * 75)
    print("      SIGNALSCOPE ROBUSTNESS TO DEGRADATION ANALYSIS (Bonus Module C)")
    print("=" * 75)
    print(f"{'Condition':30s} | {'Accuracy':10s} | {'Avg dConf':12s} | {'Flip Rate':12s}")
    print("-" * 75)
    for row in benchmarks:
        print(f"{row['Condition']:30s} | {row['Accuracy']:10s} | {row['Avg dConfidence']:12s} | {row['Verdict Flip Rate']:12s}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
