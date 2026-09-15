import os
import argparse
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

from src.model.dual_branch_net import DualBranchNet, TemperatureScaler


class ImageDataset(Dataset):
    """Dataset that loads images from real/ and ai_generated/ subdirectories."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        classes = {'real': 0, 'ai_generated': 1}
        for cls_name, label in classes.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                print(f"Warning: {cls_dir} not found, skipping.")
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(label)

        print(f"  Loaded {len(self.image_paths)} images from {root_dir}")
        real_count = self.labels.count(0)
        fake_count = self.labels.count(1)
        print(f"  Real: {real_count}, AI-Generated: {fake_count}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            # Return a blank image on error
            print(f"Error loading {img_path}: {e}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.labels[idx]
        return image, label


def get_train_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.ImageCompression(quality_lower=30, quality_upper=95, p=0.4),
        A.OneOf([
            A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
            A.CenterCrop(height=img_size, width=img_size),
        ], p=0.3),
        A.Resize(img_size, img_size),  # Ensure final size
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def save_confusion_matrix(cm, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'AI Generated'],
                yticklabels=['Real', 'AI Generated'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_history(history, output_path):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_auc'], 'g-', label='Val AUC')
    plt.plot(epochs, history['val_f1'], 'm-', label='Val F1')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_model(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load datasets
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    print("\nLoading training data...")
    train_dataset = ImageDataset(train_dir, transform=get_train_transforms(args.img_size))
    print("Loading validation data...")
    val_dataset = ImageDataset(val_dir, transform=get_val_transforms(args.img_size))

    if len(train_dataset) == 0:
        print("ERROR: No training data found! Check your data directory structure.")
        print(f"Expected: {train_dir}/real/ and {train_dir}/ai_generated/")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Build model
    print(f"\nBuilding model with backbone: {args.backbone}")
    model = DualBranchNet(num_classes=2, backbone=args.backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Separate parameter groups: backbone (lower LR) vs rest
    backbone_params = list(model.spatial_branch.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith('spatial_branch')]

    optimizer = optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': backbone_params, 'lr': args.lr / 10}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Freeze backbone for initial epochs
    print(f"Freezing backbone for first {args.freeze_backbone_epochs} epochs...")
    for param in model.spatial_branch.parameters():
        param.requires_grad = False

    best_val_auc = 0.0
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_f1': []}

    for epoch in range(args.epochs):
        # Unfreeze backbone after initial epochs
        if epoch == args.freeze_backbone_epochs:
            print("\n>> Unfreezing backbone with lower LR...")
            for param in model.spatial_branch.parameters():
                param.requires_grad = True

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # ---- Training phase ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            logits = outputs['logits']  # DualBranchNet returns dict
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)

        # ---- Validation phase ----
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                logits = outputs['logits']
                loss = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_val_loss = val_loss / len(val_dataset)
        history['val_loss'].append(epoch_val_loss)

        val_auc = roc_auc_score(all_labels, all_probs)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)

        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")

        scheduler.step()

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f">> New best model saved! AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(">> Early stopping triggered!")
                break

    # Load best model
    print("\nTraining complete. Loading best model...")
    model.load_state_dict(best_model_wts)

    # ---- Temperature Scaling ----
    print("Calibrating model using Temperature Scaling...")
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits_list.append(outputs['logits'].cpu())
            labels_list.append(labels.cpu())

    logits_tensor = torch.cat(logits_list)
    labels_tensor = torch.cat(labels_list)

    scaler = TemperatureScaler()
    optimal_temp = scaler.calibrate(logits_tensor, labels_tensor)
    scaler.save(os.path.join(args.output_dir, 'temperature.json'))
    print(f"Optimal temperature: {optimal_temp:.4f}")

    # ---- Final Metrics ----
    print("\n" + "=" * 50)
    print("FINAL METRICS ON VALIDATION SET")
    print("=" * 50)
    scaled_logits = scaler.scale(logits_tensor)
    final_probs = torch.softmax(scaled_logits, dim=1)[:, 1].numpy()
    final_preds = (final_probs > 0.5).astype(int)
    final_labels = labels_tensor.numpy()

    auc = roc_auc_score(final_labels, final_probs)
    f1 = f1_score(final_labels, final_preds, average='macro')
    acc = np.mean(final_preds == final_labels)
    cm = confusion_matrix(final_labels, final_preds)

    # False positive rate at threshold 0.5
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"ROC-AUC: {auc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"FPR @ threshold 0.5: {fpr:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:")
    print(classification_report(final_labels, final_preds, target_names=['Real', 'AI Generated']))

    save_confusion_matrix(cm, os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_history(history, os.path.join(args.output_dir, 'training_history.png'))

    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"ROC-AUC: {auc:.4f}\n")
        f.write(f"Macro-F1: {f1:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"FPR @ 0.5: {fpr:.4f}\n")
        f.write(f"Temperature: {optimal_temp:.4f}\n")
        f.write(f"Best Val AUC: {best_val_auc:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(final_labels, final_preds, target_names=['Real', 'AI Generated']))

    print(f"\nAll artifacts saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SignalScope AI-Generated Image Detector')
    parser.add_argument('--data_dir', type=str, default='data', help='Root data directory')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--output_dir', type=str, default='model/weights', help='Output directory')
    parser.add_argument('--backbone', type=str, default='efficientnet_b4', help='Backbone model')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=3, help='Epochs to freeze backbone')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    args = parser.parse_args()

    train_model(args)
