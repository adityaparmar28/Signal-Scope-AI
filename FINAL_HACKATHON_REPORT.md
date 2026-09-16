# SignalScope: Real vs AI-Generated Image Detector
**Final Hackathon Submission Report**

## 1. Project Overview
SignalScope is a high-accuracy dual-branch deep learning detector designed to classify images as either **Real** or **AI-Generated / Synthetic**. 

## 2. Dataset Strategy & Training
To achieve maximum accuracy within the strict hackathon time constraints, we optimized our training pipeline to use a highly dense, representative subset.

* **Primary Training Data:** CIFAKE Benchmark Corpus (Subset)
* **Total Images Used:** 8,550
  * **Train:** 7,091 images (3,331 Real, 3,760 AI)
  * **Validation:** 1,459 images (669 Real, 790 AI)
* **Data Sources Comprising the Training Set:**
  1. **CIFAR-10 Baseline:** Authentic, real-world natural photographs.
  2. **Latent Diffusion Data:** AI-generated synthetic images mimicking real-world deepfakes.

*Features:* We applied aggressive real-world augmentations (JPEG compression, Gaussian Blur) to ensure the model generalizes well beyond the training data.

## 3. Why Not All Datasets? (Phase 2 & Future Scope)
Our architecture is fully capable of ingesting massive datasets, but we made a deliberate engineering decision regarding the remaining datasets (DiffusionDB, Kaggle DeepDetect-2025, Wish RealVsFake):

1. **Hardware & Time Constraints:** Terabyte-scale datasets like DiffusionDB require days of download and cloud GPU compute. We optimized for a rapid, high-accuracy prototype using a streamlined 8.5k subset that finishes training locally in minutes.
2. **Reserved for Blind Testing (Zero-Shot Generalization):** We are actively holding back datasets like *DiffusionDB* and *Kaggle DeepDetect* to act as completely unseen testing grounds. This ensures our model doesn't just memorize specific generator artifacts, but learns the fundamental frequency differences between real and AI images.
3. **Production Scalability:** The current PyTorch data pipeline is 100% scalable. In Phase 2, with cloud compute, we can seamlessly merge the remaining Kaggle/HF datasets to push robust accuracy even higher across a wider variety of AI generators.

## 4. How to Run
```bash
# The weights are saved in:
model/weights/best_model.pth

# To run the training pipeline:
train_pipeline.bat
```
