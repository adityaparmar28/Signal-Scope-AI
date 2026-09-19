# SignalScope: Real vs AI-Generated Image Detector
**Final Hackathon Submission Report**

## 1. Project Overview
SignalScope is a high-accuracy dual-branch deep learning detector designed to classify images as either **Real** or **AI-Generated / Synthetic**. 

## 2. Multi-Dataset Strategy & Generator Coverage
To maximize cross-generator generalization and prevent memorization of a single generator's deconvolution fingerprints, our ingestion pipeline integrates balanced samples across diverse generative paradigms and real camera sensors:

* **1. Stable Diffusion v1.4 / CIFAKE:** Captures latent diffusion upsampling lattices.
* **2. Midjourney (v4/v5):** Photorealistic synthesis with micro-surface smoothing.
* **3. DALL-E / Diverse AI Art:** Non-photorealistic, stylistic, and hybrid generative art.
* **4. GenImage (BigGAN & GLIDE):** Traditional GAN-family synthesis vs. modern diffusion models (Bonus Module B alignment).
* **5. Authentic Photography (CIFAR-10 Photographic + RealArt):** Real-world camera captures with natural CMOS/CCD Poisson sensor noise and EXIF hardware metadata.

*Data Integrity & Deduplication:*
- **MD5 Hash Deduplication:** Strict pixel hashing guarantees zero image overlap across splits.
- **50:50 Exact Class Balancing:** Every batch contains equal representation of real and synthetic media.
- **Robust Real-World Augmentation:** Online random JPEG compression ($Q=30-90$), Gaussian blur, and perspective warps simulate social media degraded uploads.

## 3. Scalable Multi-Dataset Pipeline
Our modular ingestion engine (`src/multi_dataset_manager.py`) provides scalable streaming and thread-pooled direct downloads:

```bash
# Ingest balanced multi-generator dataset:
python src/multi_dataset_manager.py --target_cifake 3500 --target_midjourney 800 --target_diverse 600

# Execute full end-to-end training:
train_pipeline.bat
```

## 4. How to Reproduce Predictions (Section 4.1 Contract)
```bash
# Weights are preserved in:
model/weights/best_model.pth

# CLI Inference with Layer-CAM Explanation:
python model/predict.py --image data/sample_val/fake/sample_synthetic_1.jpg --explain
```
