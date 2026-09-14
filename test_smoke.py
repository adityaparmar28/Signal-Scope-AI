"""Quick smoke test — verifies all modules import and forward pass works."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 50)
print("SignalScope Smoke Test")
print("=" * 50)

# 1. SRM Filters
from src.model.srm_filters import SRMFilterLayer, SRMBranch
print("[OK] SRM filters imported")

# 2. DualBranchNet
from src.model.dual_branch_net import DualBranchNet, TemperatureScaler
print("[OK] DualBranchNet imported")

# 3. Forward pass
import torch
model = DualBranchNet(num_classes=2, backbone='efficientnet_b0', pretrained=False)
model.eval()
x = torch.randn(1, 3, 224, 224)
out = model(x)
print(f"[OK] Forward pass - logits shape: {out['logits'].shape}")

# 4. Predict module
from src.model.predict import predict_from_pil, get_transforms
print("[OK] Predict module imported")

# 5. Grad-CAM
from src.explain.gradcam import SignalScopeGradCAM
gradcam = SignalScopeGradCAM(model, device='cpu')
result = gradcam.generate(x)
print(f"[OK] Grad-CAM - heatmap shape: {result['heatmap'].shape}")

# 6. Explainer
from src.explain.explainer import generate_full_report, analyze_heatmap
analysis = analyze_heatmap(result['heatmap'])
print(f"[OK] Explainer - hotspots: {analysis['num_hotspots']}")

# 7. Temperature Scaler
scaler = TemperatureScaler()
scaled = scaler.scale(out['logits'])
print(f"[OK] Temperature scaler - scaled logits: {scaled.shape}")

# 8. Predict from PIL
from PIL import Image
import numpy as np
dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
pred = predict_from_pil(dummy_img, model, scaler, device='cpu')
print(f"[OK] Predict from PIL - verdict: {pred['verdict']}")

# 9. Full report
report = generate_full_report(pred, result['heatmap'])
print(f"[OK] Full report generated")
print(f"     Explanation: {report['explanation'][:80]}...")

print()
print("=" * 50)
print("ALL TESTS PASSED!")
print("=" * 50)
