"""
SignalScope Comprehensive Evaluation Script
Calculates official metrics required by Section 4.2 & 7.3:
- ROC-AUC (Overall & Unseen-Generator Split)
- Macro-F1 Score
- Confusion Matrix (TN, FP, FN, TP)
- Accuracy and False Positive Rate (FPR) at stated operating threshold
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.backbone import SignalScopeDetector
from src.dataset import RealVsAIDataset, get_val_transforms


def run_evaluation(model, loader, device, threshold=0.65):
    model.eval()
    all_probs = []
    all_targets = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            logits, _ = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(labels.numpy().flatten())
            all_paths.extend(paths)

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Core Metrics
    if len(np.unique(all_targets)) > 1:
        roc_auc = float(roc_auc_score(all_targets, all_probs))
    else:
        roc_auc = 0.5

    # Calculate metrics at stated fixed operating threshold (no test-set peeking leakage)
    binary_preds = (all_probs >= threshold).astype(int)
    macro_f1 = float(f1_score(all_targets, binary_preds, average="macro"))
    acc = float(accuracy_score(all_targets, binary_preds))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(all_targets, binary_preds, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Also compute at standard 0.50 baseline for comparison
    preds_50 = (all_probs >= 0.50).astype(int)
    acc_50 = float(accuracy_score(all_targets, preds_50))
    f1_50 = float(f1_score(all_targets, preds_50, average="macro"))
    tn_50, fp_50, fn_50, tp_50 = confusion_matrix(all_targets, preds_50, labels=[0, 1]).ravel()
    fpr_50 = float(fp_50 / (fp_50 + tn_50)) if (fp_50 + tn_50) > 0 else 0.0

    return {
        "roc_auc": round(roc_auc, 4),
        "operating_threshold": threshold,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix": {
            "TN (True Real)": int(tn),
            "FP (Wrongly Flagged Real)": int(fp),
            "FN (Missed Synthetic)": int(fn),
            "TP (Detected Synthetic)": int(tp)
        },
        "baseline_threshold_0_50": {
            "accuracy": round(acc_50, 4),
            "macro_f1": round(f1_50, 4),
            "false_positive_rate": round(fpr_50, 4),
            "confusion_matrix": {"TN": int(tn_50), "FP": int(fp_50), "FN": int(fn_50), "TP": int(tp_50)}
        },
        "all_probs": all_probs,
        "all_targets": all_targets,
        "all_paths": all_paths
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SignalScope Model on Held-out / Unseen Data")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test directory")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pth)")
    parser.add_argument("--threshold", type=float, default=0.65, help="Stated operating threshold")
    parser.add_argument("--unseen_dir", type=str, default=None, help="Optional separate directory for unseen generators")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignalScopeDetector().to(device)

    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"[*] Loaded weights from: {args.weights}")
    else:
        print("[!] No checkpoint provided. Evaluating with initialized baseline weights.")

    test_dataset = RealVsAIDataset(args.test_dir, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"[*] Evaluating on {len(test_dataset)} images from {args.test_dir}...")
    metrics = run_evaluation(model, test_loader, device, threshold=args.threshold)

    print("\n" + "=" * 60)
    print("           SIGNALSCOPE OFFICIAL EVALUATION REPORT")
    print("=" * 60)
    print(f" Overall ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f" Macro-F1 Score             : {metrics['macro_f1']:.4f}")
    print(f" Operating Threshold        : {metrics['operating_threshold']:.2f}")
    print(f" Test Accuracy @ Threshold  : {metrics['accuracy'] * 100:.2f}%")
    print(f" False Positive Rate (FPR)  : {metrics['false_positive_rate'] * 100:.2f}%")
    print("-" * 60)
    print(" Confusion Matrix:")
    for k, v in metrics["confusion_matrix"].items():
        print(f"   {k:28s} : {v}")

    # If unseen-generator split is provided separately
    if args.unseen_dir and os.path.exists(args.unseen_dir):
        unseen_dataset = RealVsAIDataset(args.unseen_dir, transform=get_val_transforms())
        unseen_loader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
        unseen_metrics = run_evaluation(model, unseen_loader, device, threshold=args.threshold)
        print("-" * 60)
        print(f" [!] UNSEEN-GENERATOR SPLIT ROC-AUC : {unseen_metrics['roc_auc']:.4f}")
        print(f"     Unseen Macro-F1               : {unseen_metrics['macro_f1']:.4f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
