"""
SignalScope Model Training Script (v2 - Multi-Generator)
Trains the dual-branch detector with:
- AMP mixed-precision for 2x faster GPU training
- Early stopping with patience=4
- OneCycleLR scheduler (better warmup for fine-tuning)
- ROC-AUC and Macro-F1 monitoring
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, f1_score

# Ensure model package is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.backbone import SignalScopeDetector
from src.dataset import RealVsAIDataset, get_training_transforms, get_val_transforms


from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, epoch=1, total_epochs=10):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits, _ = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(loader.dataset)


def evaluate_epoch(model, loader, device, epoch=1, total_epochs=10):
    model.eval()
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Valid]", leave=False)
    with torch.no_grad():
        for images, labels, _ in pbar:
            images = images.to(device)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits, _ = model(images)
            probs = torch.sigmoid(logits.float()).cpu().numpy().flatten()
            all_preds.extend(probs)
            all_targets.extend(labels.numpy().flatten())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_preds)
        binary_preds = (all_preds >= 0.5).astype(int)
        macro_f1 = f1_score(all_targets, binary_preds, average="macro")
    else:
        auc = 0.5
        macro_f1 = 0.5

    return auc, macro_f1


def main():
    parser = argparse.ArgumentParser(description="Train SignalScope Real vs Synthetic Detector")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory with train/ and val/")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=3e-5, help="Peak learning rate")
    parser.add_argument("--weights_dir", type=str, default="model/weights", help="Directory to save checkpoints")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    train_path = os.path.join(args.data_dir, "train")
    val_path = os.path.join(args.data_dir, "val")

    if not os.path.exists(train_path):
        print(f"[!] Warning: Train directory {train_path} not found. Please populate dataset.")
        return

    train_dataset = RealVsAIDataset(train_path, transform=get_training_transforms())
    val_dataset = RealVsAIDataset(val_path, transform=get_val_transforms())

    print(f"[*] Dataset sizes:")
    print(f"    Train: {len(train_dataset)} images")
    print(f"    Val:   {len(val_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = SignalScopeDetector().to(device)

    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"[*] Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # OneCycleLR: warm up then decay — better than cosine for fine-tuning
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos'
    )

    # AMP scaler for mixed precision
    scaler = GradScaler('cuda') if device.type == "cuda" else None

    os.makedirs(args.weights_dir, exist_ok=True)
    best_auc = 0.0
    epochs_without_improvement = 0

    print(f"[*] Starting training for {args.epochs} epochs (patience={args.patience})...")
    print(f"    LR: {args.lr}, Batch: {args.batch_size}, AMP: {device.type == 'cuda'}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler, epoch=epoch, total_epochs=args.epochs)
        val_auc, val_f1 = evaluate_epoch(model, val_loader, device, epoch=epoch, total_epochs=args.epochs)
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {train_loss:.4f} | "
              f"AUC: {val_auc:.4f} | F1: {val_f1:.4f} | "
              f"LR: {lr_now:.2e} | {elapsed:.0f}s")

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            save_path = os.path.join(args.weights_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f" -> [BEST] Saved to {save_path} (AUC: {best_auc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\n[*] Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Save final model too
    final_path = os.path.join(args.weights_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)

    print("-" * 75)
    print(f"[*] Training complete! Best Val AUC: {best_auc:.4f}")
    print(f"    Best model: {os.path.join(args.weights_dir, 'best_model.pth')}")
    print(f"    Final model: {final_path}")


if __name__ == "__main__":
    main()
