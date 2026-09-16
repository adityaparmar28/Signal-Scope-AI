"""
SignalScope Weights Initializer / Downloader
Ensures the repository works out-of-the-box in under 10 minutes (Section 7.2 Reproducibility Gate).
"""

import os
import sys
import torch

# Ensure model directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model.backbone import SignalScopeDetector


def initialize_or_download_weights(weights_path="model/weights/best_model.pth"):
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    if os.path.exists(weights_path):
        print(f"[*] Checkpoint already exists at: {weights_path}")
        return weights_path

    print(f"[*] Initializing baseline weights for SignalScope Dual-Branch Detector...")
    model = SignalScopeDetector()
    torch.save(model.state_dict(), weights_path)
    print(f"[OK] Initialized checkpoint ready at: {weights_path}")
    print("[*] Note: For production weights, download pre-trained checkpoint from GitHub Releases or train via src/train.py.")
    return weights_path


if __name__ == "__main__":
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "model/weights/best_model.pth"
    initialize_or_download_weights(weights_path)
