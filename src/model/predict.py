import os
import argparse
import torch
import numpy as np
from PIL import Image
from src.model.dual_branch_net import DualBranchNet, TemperatureScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Module-level cache for model
_CACHE = {
    'model': None,
    'scaler': None,
    'device': None,
    'model_dir': None
}


def get_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def _load_model_cached(model_dir, device):
    """Load model with caching to avoid reloading on every prediction."""
    global _CACHE
    if _CACHE['model'] is not None and _CACHE['model_dir'] == model_dir:
        return _CACHE['model'], _CACHE['scaler']

    model_path = os.path.join(model_dir, 'best_model.pth')

    model = DualBranchNet(num_classes=2, backbone='efficientnet_b4', pretrained=False)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    else:
        print(f"WARNING: {model_path} not found. Using uninitialized weights.")

    model.to(device)
    model.eval()

    scaler = TemperatureScaler()
    temp_path = os.path.join(model_dir, 'temperature.json')
    scaler.load(temp_path)

    _CACHE['model'] = model
    _CACHE['scaler'] = scaler
    _CACHE['device'] = device
    _CACHE['model_dir'] = model_dir

    return model, scaler


def predict(image_path, model_dir='model/weights', device=None):
    """
    Predict whether an image is real or AI-generated.

    Args:
        image_path: Path to the image file
        model_dir: Directory containing model weights and temperature.json
        device: torch device (auto-detected if None)

    Returns:
        dict with keys: label, confidence, raw_logits, verdict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, scaler = _load_model_cached(model_dir, device)

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {"error": f"Failed to load image: {e}"}

    transform = get_transforms()
    img_np = np.array(image)
    tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        raw_logits = outputs['logits']
        calibrated_logits = scaler.scale(raw_logits)
        probs = torch.softmax(calibrated_logits, dim=1)[0]

    fake_prob = probs[1].item()
    real_prob = probs[0].item()

    is_fake = fake_prob > 0.5
    confidence = fake_prob if is_fake else real_prob
    label = 'ai_generated' if is_fake else 'real'

    verdict_text = "Likely AI-Generated" if is_fake else "Likely Real"
    verdict = f"{verdict_text} (confidence: {confidence:.2f})"

    return {
        'label': label,
        'confidence': float(confidence),
        'raw_logits': raw_logits[0].cpu().numpy().tolist(),
        'verdict': verdict,
        'fake_probability': float(fake_prob)
    }


def predict_from_pil(image_pil, model, scaler, device='cpu'):
    """Predict from a PIL Image object directly (used by Streamlit app)."""
    transform = get_transforms()
    img_np = np.array(image_pil.convert('RGB'))
    tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        raw_logits = outputs['logits']
        calibrated_logits = scaler.scale(raw_logits)
        probs = torch.softmax(calibrated_logits, dim=1)[0]

    fake_prob = probs[1].item()
    real_prob = probs[0].item()

    is_fake = fake_prob > 0.5
    confidence = fake_prob if is_fake else real_prob
    label = 'ai_generated' if is_fake else 'real'

    verdict_text = "Likely AI-Generated" if is_fake else "Likely Real"
    verdict = f"{verdict_text} (confidence: {confidence:.2f})"

    return {
        'label': label,
        'confidence': float(confidence),
        'raw_logits': raw_logits[0].cpu().numpy().tolist(),
        'verdict': verdict,
        'fake_probability': float(fake_prob),
        'tensor': tensor  # Keep tensor for Grad-CAM
    }


def batch_predict(image_paths, model_dir='model/weights', device=None):
    """Predict on multiple images."""
    return [predict(path, model_dir, device) for path in image_paths]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SignalScope Prediction')
    parser.add_argument('--image', type=str, required=True, help="Path to image file")
    parser.add_argument('--model_dir', type=str, default='model/weights', help="Path to model weights")
    args = parser.parse_args()

    result = predict(args.image, args.model_dir)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\n{'='*50}")
        print(f"  {result['verdict']}")
        print(f"{'='*50}")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  AI-Generated Probability: {result['fake_probability']:.4f}")
