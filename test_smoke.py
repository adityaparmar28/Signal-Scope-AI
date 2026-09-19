"""
SignalScope Comprehensive Smoke & Integration Test Suite
Verifies all modules, forward pass, calibration, Layer-CAM, and Predict interface.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("      SIGNALSCOPE SYSTEM SMOKE & INTEGRATION TEST")
print("=" * 60)

# 1. Test Core SignalScopeDetector v2 Backbone
from model.backbone import SignalScopeDetector, SRMConv2d
print("[OK] 1. SignalScopeDetector v2 & SRM Conv2d imported successfully.")

import torch
model = SignalScopeDetector()
model.eval()
x = torch.randn(1, 3, 224, 224)
logit, feat_map = model(x)
print(f"[OK] 2. Forward pass verified - Logit: {logit.shape}, FeatureMap: {feat_map.shape}")

# 2. Test Probability Calibration & Responsible Verdicts
from model.calibration import TemperatureCalibrator
calibrator = TemperatureCalibrator(temperature=1.15)
verdict_ai = calibrator.get_verdict(0.92)
verdict_real = calibrator.get_verdict(0.08)
verdict_mid = calibrator.get_verdict(0.50)
print(f"[OK] 3. Temperature Calibration verified - Verdict AI: '{verdict_ai['verdict']}', Real: '{verdict_real['verdict']}'")

# 3. Test Metadata & C2PA Provenance Inspector
from model.metadata_inspector import MetadataInspector
inspector = MetadataInspector()
print("[OK] 4. MetadataInspector & C2PA manifest scanner initialized.")

# 4. Test Layer-CAM Explainer
from model.explainer import LayerCAMExplainer
from PIL import Image
import numpy as np

explainer = LayerCAMExplainer(model)
dummy_pil = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
dummy_tensor = torch.randn(1, 3, 224, 224)
explanation = explainer.explain_image(dummy_pil, dummy_tensor)
print(f"[OK] 5. Layer-CAM Explainer verified - Cues extracted: {len(explanation['visual_cues'])}")

# 5. Check Checkpoint Weights Availability
weights_candidates = [
    "model/weights/best_model.pth",
    "model/signalscope_model.pth"
]
found_weights = [w for w in weights_candidates if os.path.exists(w)]
if found_weights:
    print(f"[OK] 6. Model checkpoint verified at: {found_weights[0]} ({os.path.getsize(found_weights[0]) / (1024*1024):.1f} MB)")
else:
    print("[!] 6. Note: Model weights will be populated upon running training.")

print()
print("=" * 60)
print("   ALL SIGNALSCOPE SYSTEM SMOKE TESTS PASSED CLEANLY!  ")
print("=" * 60)
