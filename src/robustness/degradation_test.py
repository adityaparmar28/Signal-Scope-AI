"""
Robustness Analysis: Test model performance under various image degradations.

Tests: JPEG compression, resizing, Gaussian noise, screenshot simulation.
Outputs: CSV results table + degradation-vs-AUC plot + summary report.

Usage:
  python -m src.robustness.degradation_test --model_dir model/weights --test_dir data/test --output_dir report/robustness
"""
import os
import sys
import argparse
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---- Degradation Functions ----

def jpeg_compress(image_pil, quality):
    buf = io.BytesIO()
    image_pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def resize_degrade(image_pil, scale):
    w, h = image_pil.size
    small = image_pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def add_gaussian_noise(image_pil, std):
    img_np = np.array(image_pil).astype(np.float32)
    noise = np.random.normal(0, std, img_np.shape)
    noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def screenshot_simulate(image_pil, _param=None):
    img = jpeg_compress(image_pil, 85)
    return resize_degrade(img, 0.9)


def fgsm_attack(image_pil, param_dict):
    """
    Applies FGSM attack to fool the model.
    param_dict must contain: 'epsilon', 'model', 'device'
    """
    import torch
    from src.model.predict import get_transforms
    
    epsilon = param_dict['epsilon']
    model = param_dict['model']
    device = param_dict['device']
    
    transform = get_transforms()
    img_np = np.array(image_pil.convert('RGB'))
    tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)
    
    tensor.requires_grad = True
    
    outputs = model(tensor)
    logits = outputs['logits']
    pred = logits.argmax(dim=1)
    
    loss = torch.nn.CrossEntropyLoss()(logits, pred)
    
    model.zero_grad()
    loss.backward()
    
    data_grad = tensor.grad.data
    perturbed_tensor = tensor + epsilon * data_grad.sign()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    perturbed_image = perturbed_tensor.squeeze(0) * std + mean
    perturbed_image = perturbed_image.clamp(0, 1)
    
    perturbed_np = (perturbed_image.cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(perturbed_np)

# ---- Main Analysis ----

def collect_test_images(test_dir, max_samples=500):
    """Collect image paths and labels from test directory."""
    images = []
    labels = []
    label_map = {'real': 0, 'ai_generated': 1}

    for cls_name, label in label_map.items():
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        fnames = sorted(os.listdir(cls_dir))
        for fname in fnames:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                images.append(os.path.join(cls_dir, fname))
                labels.append(label)

    # Subsample if needed
    if max_samples and len(images) > max_samples:
        indices = np.random.choice(len(images), max_samples, replace=False)
        images = [images[i] for i in indices]
        labels = [labels[i] for i in indices]

    return images, labels


def run_robustness_analysis(model_dir, test_dir, output_dir, device='cpu', max_samples=500):
    """Run full robustness analysis."""
    import torch
    from src.model.predict import predict_from_pil, get_transforms
    from src.model.dual_branch_net import DualBranchNet, TemperatureScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)

    # Load model
    print("Loading model...")
    model_path = os.path.join(model_dir, 'best_model.pth')
    model = DualBranchNet(num_classes=2, backbone='efficientnet_b4', pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    scaler = TemperatureScaler()
    scaler.load(os.path.join(model_dir, 'temperature.json'))

    # Collect test images
    print(f"Collecting test images from {test_dir}...")
    image_paths, true_labels = collect_test_images(test_dir, max_samples)
    print(f"Found {len(image_paths)} test images")

    if len(image_paths) == 0:
        print("ERROR: No test images found!")
        return

    # Define degradations
    degradations = {
        'jpeg_q10': (jpeg_compress, 10),
        'jpeg_q30': (jpeg_compress, 30),
        'jpeg_q50': (jpeg_compress, 50),
        'jpeg_q70': (jpeg_compress, 70),
        'jpeg_q90': (jpeg_compress, 90),
        'resize_0.25': (resize_degrade, 0.25),
        'resize_0.50': (resize_degrade, 0.50),
        'resize_0.75': (resize_degrade, 0.75),
        'noise_std5': (add_gaussian_noise, 5),
        'noise_std15': (add_gaussian_noise, 15),
        'noise_std25': (add_gaussian_noise, 25),
        'screenshot': (screenshot_simulate, None),
        'fgsm_eps0.01': (fgsm_attack, {'epsilon': 0.01, 'model': model, 'device': device}),
        'fgsm_eps0.05': (fgsm_attack, {'epsilon': 0.05, 'model': model, 'device': device}),
        'none': (None, None),  # Baseline - no degradation
    }

    results = []

    for deg_name, (deg_func, deg_param) in degradations.items():
        print(f"\nTesting: {deg_name}...")
        preds = []
        probs = []

        for img_path in tqdm(image_paths, desc=deg_name):
            try:
                img = Image.open(img_path).convert('RGB')

                # Apply degradation
                if deg_func is not None:
                    img = deg_func(img, deg_param)

                # Predict
                result = predict_from_pil(img, model, scaler, device=device)
                probs.append(result['fake_probability'])
                preds.append(1 if result['label'] == 'ai_generated' else 0)

            except Exception as e:
                print(f"  Error on {img_path}: {e}")
                probs.append(0.5)
                preds.append(0)

        # Compute metrics
        try:
            auc = roc_auc_score(true_labels, probs)
        except ValueError:
            auc = 0.5
        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average='macro')

        results.append({
            'degradation': deg_name,
            'auc': round(auc, 4),
            'accuracy': round(acc, 4),
            'f1': round(f1, 4)
        })
        print(f"  AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'robustness_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # JPEG plot
    jpeg_df = df[df['degradation'].str.startswith('jpeg')]
    if not jpeg_df.empty:
        qualities = [int(d.split('_q')[1]) for d in jpeg_df['degradation']]
        axes[0].plot(qualities, jpeg_df['auc'], 'bo-', label='AUC', linewidth=2)
        axes[0].set_xlabel('JPEG Quality')
        axes[0].set_ylabel('Score')
        axes[0].set_title('JPEG Compression Robustness')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0.4, 1.05)

    # Resize plot
    resize_df = df[df['degradation'].str.startswith('resize')]
    if not resize_df.empty:
        scales = [float(d.split('_')[1]) for d in resize_df['degradation']]
        axes[1].plot(scales, resize_df['auc'], 'ro-', label='AUC', linewidth=2)
        axes[1].set_xlabel('Resize Scale Factor')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Resize Degradation Robustness')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0.4, 1.05)

    # Noise plot
    noise_df = df[df['degradation'].str.startswith('noise')]
    if not noise_df.empty:
        stds = [int(d.split('std')[1]) for d in noise_df['degradation']]
        axes[2].plot(stds, noise_df['auc'], 'go-', label='AUC', linewidth=2)
        axes[2].set_xlabel('Noise Std Dev')
        axes[2].set_ylabel('Score')
        axes[2].set_title('Gaussian Noise Robustness')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0.4, 1.05)

    plt.suptitle('SignalScope Robustness Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'robustness_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")

    # ---- Summary Report ----
    baseline = df[df['degradation'] == 'none'].iloc[0] if 'none' in df['degradation'].values else None
    report_path = os.path.join(output_dir, 'robustness_report.txt')
    with open(report_path, 'w') as f:
        f.write("SignalScope Robustness Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        if baseline is not None:
            f.write(f"Baseline (no degradation): AUC={baseline['auc']:.4f}, Acc={baseline['accuracy']:.4f}\n\n")
        f.write("Results by degradation:\n")
        f.write("-" * 50 + "\n")
        for _, row in df.iterrows():
            delta = ""
            if baseline is not None and row['degradation'] != 'none':
                d = row['auc'] - baseline['auc']
                delta = f" (Δ AUC: {d:+.4f})"
            f.write(f"  {row['degradation']:20s} | AUC: {row['auc']:.4f} | Acc: {row['accuracy']:.4f} | F1: {row['f1']:.4f}{delta}\n")
        f.write("\n\nConclusion:\n")
        f.write("The model's robustness to various degradations is summarized above.\n")
        f.write("Lower scores under degradation indicate areas for improvement.\n")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SignalScope Robustness Analysis')
    parser.add_argument('--model_dir', type=str, required=True, help='Model weights directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_samples', type=int, default=500, help='Max test samples')
    args = parser.parse_args()

    run_robustness_analysis(args.model_dir, args.test_dir, args.output_dir, args.device, args.max_samples)
