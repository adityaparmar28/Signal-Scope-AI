# 🔍 SignalScope

**Telling Real From Synthetic in the Age of Generative Media**

> SIH 2026 Internal Hackathon | L.J. Institute of Engineering and Technology [C-433] | Problem Statement 2

---

## 📋 Modules Built

| Module | Status | Description |
|--------|--------|-------------|
| **Core: Real vs AI-Generated Classification** | ✅ Built | Binary classifier with calibrated confidence |
| **Module A: Faithful Explanation** | ✅ Built | Grad-CAM heatmaps + template-based text explanations |
| **Module B: Generator Attribution** | 🔧 Architecture Ready | Multi-class head for GAN vs Diffusion attribution |
| **Module C: Robustness to Degradation** | ✅ Built | JPEG, resize, noise, screenshot robustness analysis |
| **Module D: Provenance & Metadata** | ✅ Built | EXIF/C2PA signature detection integrated into UI |
| **Module F: Deployable Interface** | ✅ Built | Streamlit drag-and-drop web application |
| **Module G: Active Defence Analysis** | ✅ Built | FGSM adversarial attack testing in robustness suite |

---

## 🚀 Quick Start (Setup & Run)

### Prerequisites
- Python 3.10+
- pip

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python -m data.download_data --output_dir data --max_samples 60000
```

### 3. Train the Model
```bash
# GPU (recommended)
python -m src.model.train --data_dir data --epochs 15 --batch_size 32 --backbone efficientnet_b4

# CPU (slower, reduce batch size)
python -m src.model.train --data_dir data --epochs 10 --batch_size 16 --backbone efficientnet_b0
```

### 4. Run a Prediction
```bash
python -m src.model.predict --image path/to/image.jpg --model_dir model/weights
```

### 5. Launch ZeroGPT-Grade Web UI
```bash
streamlit run app/app.py
```

### 6. Run Robustness Analysis
```bash
python src/degradation_benchmark.py
```

> ⏱️ A judge should be able to reproduce a prediction in under 10 minutes by running steps 1, then 4.

---

## 📊 Datasets Used

| Dataset | Source | Generator Family | Images |
|---------|--------|------------------|--------|
| **Midjourney Dataset** | [HuggingFace (ehristoforu)](https://huggingface.co/datasets/ehristoforu/midjourney-images) | Midjourney v4 / v5 | 500+ |
| **Diverse AI Art & Real** | [HuggingFace (Hemg)](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets) | DALL-E / Generative Art & Real | 800+ |
| **CIFAKE** | [HuggingFace (dragonintelligence)](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset) | Stable Diffusion v1.4 | 7,200 |
| **ImageNet-1K (via EfficientNet)** | Pretrained backbone weights (`timm`) | Natural photography | 1.4M (Pretrained) |

---

## 📈 Reported Metrics (Multi-Generator Evaluation)

*Evaluated on 1,459 held-out images covering Midjourney, AI Art, Stable Diffusion, and Real Photos:*

| Metric | Operating Threshold (0.65) | Balanced Threshold (0.50) |
|--------|----------------------------|----------------------------|
| **Overall ROC-AUC** | **0.9732 (97.32%)** | **0.9732 (97.32%)** |
| **Unseen-Generator-Split AUC** | **0.9685 (96.85%)** | **0.9685 (96.85%)** |
| **Macro-F1 Score** | **0.8461** | **0.9015** |
| **Test Accuracy** | **84.65%** (1,235 / 1,459) | **90.75%** (1,324 / 1,459) |
| **False Positive Rate (FPR)** | **2.24%** (only 15 misflagged real photos) | **4.48%** (30 misflagged real photos) |
| **Calibration Temperature** | **1.1500** | **1.1500** |

### Confusion Matrix Breakdown (Operating Threshold 0.65)
- **True Real (TN)**: 654
- **False Positive (FP)**: 15 *(Strictly controlled false alarm rate of 2.24%)*
- **Missed Synthetic (FN)**: 209
- **Detected Synthetic (TP)**: 581

> 📊 Confusion matrix visualization: `report/confusion_matrices.png`  
> 📈 ROC & Degradation curves: `report/evaluation_metrics.png`  
> 📄 Official model report: `report/model_report.md`

---

## 🏗️ Architecture Overview

### Dual-Branch Architecture

```
Input Image (224×224)
        │
        ├──────────────────┐
        ▼                  ▼
┌─────────────────┐ ┌──────────────────┐
│  Spatial Branch  │ │  Frequency Branch │
│  EfficientNet-B4 │ │  SRM High-Pass    │
│  (ImageNet pre-  │ │  Filters → CNN    │
│   trained)       │ │  (3 SRM kernels)  │
└────────┬────────┘ └────────┬─────────┘
         │                   │
         └────────┬──────────┘
                  ▼
         ┌──────────────┐
         │ Feature Fusion│
         │ FC → BN → ReLU│
         │ → Dropout     │
         └──────┬───────┘
                │
         ┌──────┴───────┐
         ▼              ▼
    ┌─────────┐  ┌────────────┐
    │ Binary  │  │ Attribution│
    │Classifier│  │   Head    │
    │(real/AI) │  │(GAN/Diff) │
    └────┬────┘  └───────────┘
         ▼
  Temperature Scaling
         ▼
  Calibrated Verdict
         ▼
  Grad-CAM Explainer
         ▼
    Streamlit UI
```

### Why Dual-Branch?
- **Spatial branch** (EfficientNet-B4): Detects visual artifacts — texture inconsistencies, geometry errors, lighting/shadow issues
- **Frequency branch** (SRM filters): Detects spectral fingerprints left by generators — these are **generator-agnostic** and help generalize to unseen generators
- **Feature fusion**: Combines both signal types for robust classification

### Robustness & Calibration
- **Training augmentation**: JPEG compression (q30-95), random resize/crop, Gaussian noise, brightness/contrast — simulates real-world degradation
- **Temperature scaling**: Post-hoc calibration on validation set ensures confidence scores are honest and well-calibrated
- **Backbone freezing**: First 3 epochs freeze EfficientNet backbone, then unfreeze with 10× lower LR — prevents catastrophic forgetting

### Explainability
- **Grad-CAM**: Generates spatial heatmaps highlighting regions the model focuses on
- **Template-based explanation**: Analyzes heatmap regions and generates faithful, hedged natural-language explanations
- Explanations cite specific image regions and use probabilistic language ("likely", "suggests")

---

## 🔒 Known Limitations

1. **CIFAKE images are 32×32**: Low resolution limits fine-grained artifact detection. Models trained on higher-resolution datasets (GenImage, DiffusionDB) would perform better.
2. **Unseen generators**: Performance may degrade on generators very different from training data (e.g., very new models).
3. **Adversarial robustness**: The model has not been hardened against deliberate adversarial attacks.
4. **Text in images**: The model does not specifically detect text rendering artifacts (a common AI tell).
5. **Calibration drift**: Temperature scaling is calibrated on the validation set; distribution shift may affect calibration on novel data.

---

## 📁 Repository Structure

```
signalscope/
├── README.md                           # This file
├── requirements.txt                    # Dependencies
├── src/
│   ├── model/
│   │   ├── srm_filters.py             # SRM high-pass frequency filters
│   │   ├── dual_branch_net.py         # Dual-branch architecture + TemperatureScaler
│   │   ├── train.py                   # Training script with augmentation
│   │   └── predict.py                 # Prediction interface (CLI + API)
│   ├── explain/
│   │   ├── gradcam.py                 # Grad-CAM heatmap generation
│   │   └── explainer.py              # Faithful text explanation generator
│   ├── robustness/
│   │   └── degradation_test.py       # Robustness analysis under degradation
│   └── app/
│       └── streamlit_app.py          # Streamlit web interface
├── model/
│   └── weights/                       # Trained model weights
├── data/
│   └── download_data.py              # Dataset download script
└── report/
    └── model_report.md               # One-page model report
```

---

## 🎬 Demo Video

[Link to demo video — *to be added*]

---

## 📜 Originality Declaration

- **Architecture**: Custom dual-branch design combining EfficientNet (from `timm` library) with SRM steganalysis filters
- **Libraries**: PyTorch, timm, pytorch-grad-cam, albumentations, Streamlit, scikit-learn
- **Pretrained weights**: EfficientNet-B4 pretrained on ImageNet (from `timm`)
- **Dataset**: CIFAKE (MIT license)
- **AI assistants**: Used for code scaffolding; the working system and evaluation are our own

All third-party code is properly attributed. No public real-vs-fake notebooks were copied wholesale.
