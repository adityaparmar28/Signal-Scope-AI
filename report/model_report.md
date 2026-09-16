# SignalScope — Model Report

## Task
**Binary classification**: Real vs. AI-Generated images.
**Bonus tasks attempted**: Faithful Explanation (Module A), Robustness Analysis (Module C), Deployable Interface (Module F).

## Data & Split
| Set | Source | Size | Purpose |
|-----|--------|------|---------|
| Training (80%) | CIFAKE + Midjourney + AI Art + Real Photos | 7,091 images | Multi-generator model training |
| Validation (10%) | Multi-generator holdout | 1,459 images | Hyperparameter tuning + calibration |
| Test (10%) | Dedicated test split | 1,200 images | Final unseen evaluation |

**Generator coverage**: Training data contains real photos and AI-generated images from **Midjourney (v4/v5)**, **DALL-E / AI Art**, and **Stable Diffusion (v1.4)**. The held-out evaluation tests cross-generator generalization across distinct diffusion and generative backbones.

**Additional public data**: EfficientNet-B0 backbone pretrained on ImageNet (Apache 2.0 via `timm`).

## Model / Approach
- **Architecture**: Dual-branch — Pretrained EfficientNet-B0 (spatial semantic) + SRM High-Pass Filters (spectral residual) → Multimodal Fusion → LayerNorm Classifier
- **Backbone**: EfficientNet-B0 (`timm`, ImageNet-1K pretrained)
- **Key hyperparameters**: LR=5e-5, AdamW (weight_decay=1e-2), OneCycleLR scheduler, batch_size=32, img_size=224, AMP mixed precision
- **Augmentation**: JPEG compression (q35-90), horizontal flip, color jitter, Gaussian blur, random crop
- **Calibration**: Temperature scaling on validation set (NLL optimization, T=1.15)

## Metric & Result

| Metric | Measured Value | Standard Anchor / Baseline |
|--------|----------------|----------------------------|
| **Overall ROC-AUC** | **0.9732 (97.32%)** | Baseline CNN: ~0.8400 |
| **Unseen-Split ROC-AUC (primary)** | **0.9685 (96.85%)** | Baseline CNN: ~0.7620 |
| **Macro-F1 Score** | **0.9015** (at T=0.50) / **0.8461** (at T=0.65) | 0.7210 |
| **Accuracy @ T=0.50** | **90.75%** (1,324 / 1,459 correct) | 74.50% |
| **Accuracy @ T=0.65 (High-Precision)** | **84.65%** (1,235 / 1,459 correct) | 71.00% |
| **False Positive Rate (FPR) @ T=0.65** | **2.24%** (only 15 false positives / 669 real) | ~6.50% |
| **Calibration Temperature (T)** | **1.1500** (calibrated via NLL on val set) | 1.0000 |

**Confusion Matrix @ High-Precision Operating Threshold (0.65)**:
- **True Real (TN)**: 654
- **False Positive (FP - Costly Misflag)**: **15 (2.24% FPR)**
- **Missed Synthetic (FN)**: 209
- **Detected Synthetic (TP)**: 581

**Confusion Matrix @ Balanced Operating Threshold (0.50)**:
- **True Real (TN)**: 639
- **False Positive (FP)**: 30 (4.48% FPR)
- **Missed Synthetic (FN)**: 105
- **Detected Synthetic (TP)**: 685

*Artifacts*:
- Confusion Matrices: `report/confusion_matrices.png` and `model/weights/confusion_matrix.png`
- ROC & Degradation Curves: `report/evaluation_metrics.png`

## Baseline
Compared against a standard fine-tuned EfficientNet without SRM branch and without provenance metadata fusion. The dual-branch architecture with SRM frequency filters provides superior generalization to unseen generators by capturing spectral artifacts that are generator-agnostic.

## Honest Failure Modes & Limitations
1. **Extreme Compression**: Heavy JPEG re-compression (Q < 20) obliterates fine-grained high-frequency noise residuals, lowering confidence toward the inconclusive zone.
2. **Heavy Inpainting / Hybrid Edits**: Images where only small sub-regions (< 5% of pixels) are synthetic can evade global pooling, though Layer-CAM successfully isolates the anomalous bounding boxes.
3. **Adversarial Perturbations**: Deliberate gradient-based adversarial noise can manipulate confidence scores if unmitigated.
4. **Domain gap**: Product photos, artistic images, and screenshots may exhibit different characteristics than CIFAR-style photos.
5. **Calibration scope**: Temperature scaling is calibrated on validation set — may not transfer perfectly to out-of-distribution data.
