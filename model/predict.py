"""
SignalScope Official Predict Interface
Fulfills the mandatory evaluation contract (Section 3.1 & 7.1).
Accepts a single image and outputs a responsible likelihood verdict,
calibrated confidence score, and optional faithful explanation.
"""

import argparse
import json
import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image

# Ensure local model package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.backbone import SignalScopeDetector
from model.calibration import TemperatureCalibrator
from model.explainer import LayerCAMExplainer
from model.metadata_inspector import MetadataInspector


def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model(weights_path=None, device="cpu"):
    model = SignalScopeDetector()
    if weights_path is None:
        candidates = [
            os.path.join(parent_dir, "model", "weights", "best_model.pth"),
            os.path.join(parent_dir, "model", "signalscope_model.pth"),
            os.path.join(current_dir, "weights", "best_model.pth"),
            os.path.join(current_dir, "signalscope_model.pth"),
            "model/weights/best_model.pth",
            "model/signalscope_model.pth"
        ]
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                break

    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"[*] Loaded SignalScope weights from: {weights_path}")
        except Exception as e:
            print(f"[Warning] Could not load weights from {weights_path}: {e}. Running with initialized weights.")
    else:
        print("[!] No checkpoint found. Running with initialized baseline weights.")

    model.to(device)
    model.eval()
    return model


def predict(image_path: str, weights_path: str = None, explain: bool = False, output_dir: str = "output"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, device=device)
    calibrator = TemperatureCalibrator(temperature=1.15)
    inspector = MetadataInspector()

    # Load and transform image
    pil_image = Image.open(image_path).convert("RGB")
    transform = get_inference_transforms()
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Core detection
    with torch.no_grad():
        logit, _ = model(input_tensor)
        logit_val = logit.item()

    # Probability calibration (honest uncertainty)
    calibrated_prob = calibrator.calibrate_probability(logit_val)

    # Provenance & Metadata inspection (Module D)
    metadata_info = inspector.inspect_file(image_path)
    fused_assessment = inspector.fuse_signals(calibrated_prob, metadata_info)
    fused_prob = fused_assessment["fused_probability"]

    # Final verdict and confidence derived from fused evidence
    final_verdict_info = calibrator.get_verdict(fused_prob)

    result = {
        "file": os.path.basename(image_path),
        "verdict": final_verdict_info["verdict"],
        "calibrated_confidence": final_verdict_info["display_confidence"],
        "raw_probability_ai": round(calibrated_prob, 4),
        "fused_probability_ai": fused_prob,
        "recommendation": final_verdict_info["recommendation"],
        "metadata_signal": metadata_info["metadata_summary"],
    }

    # Optional Headline Bonus A: Faithful Explainability
    if explain:
        os.makedirs(output_dir, exist_ok=True)
        explainer = LayerCAMExplainer(model)
        # Explainer requires grad enabled on input tensor
        explanation_data = explainer.explain_image(pil_image, input_tensor.clone())

        overlay_filename = f"explanation_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
        overlay_path = os.path.join(output_dir, overlay_filename)
        Image.fromarray(explanation_data["heatmap_overlay"]).save(overlay_path)

        result["explanation"] = {
            "cues": explanation_data["visual_cues"],
            "summary": explanation_data["faithful_explanation"],
            "heatmap_path": overlay_path,
            "anomalous_regions_count": len(explanation_data["anomaly_boxes"])
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="SignalScope AI Media Forensics Predict Interface")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default=None, help="Path to model checkpoint (.pth)")
    parser.add_argument("--explain", action="store_true", help="Generate Layer-CAM heatmap and grounded explanation")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save explanation visuals")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    try:
        results = predict(
            image_path=args.image,
            weights_path=args.weights,
            explain=args.explain,
            output_dir=args.output_dir
        )

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("\n" + "=" * 50)
            print("      SIGNALSCOPE - IMAGE CHECK RESULT")
            print("=" * 50)
            print(f" Image          : {results['file']}")
            print(f" Result         : {results['verdict']}")
            print(f" Confidence     : {results['calibrated_confidence'] * 100:.1f}%")
            print(f" AI Score       : {results['fused_probability_ai'] * 100:.1f}%")
            print(f" Suggestion     : {results['recommendation']}")
            print(f" File Info      : {results['metadata_signal']}")
            if "explanation" in results:
                print("-" * 50)
                print(" WHY WE THINK THIS:")
                print(f" Clues Found    : {', '.join(results['explanation']['cues'])}")
                print(f" Explanation    : {results['explanation']['summary']}")
                print(f" Heatmap Saved  : {results['explanation']['heatmap_path']}")
            print("=" * 50 + "\n")

    except Exception as e:
        print(f"[Error] Prediction failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
