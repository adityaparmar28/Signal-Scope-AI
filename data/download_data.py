"""
Download and extract CIFAKE dataset for SignalScope training and evaluation.
Uses HuggingFace 'dragonintelligence/CIFAKE-image-dataset' (already cached locally).
Supports train, validation, and test splits with real/ and fake/ directories.
"""
import os
import sys
import argparse
import random
from PIL import Image
from tqdm import tqdm


def prepare_cifake(output_dir="data", max_train=8000, max_val=2000, max_test=2000, seed=42):
    random.seed(seed)
    from datasets import load_dataset

    print("=" * 60)
    print("Preparing CIFAKE dataset from HuggingFace cache...")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake", "ai_generated"]:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

    print("[*] Loading train split from dragonintelligence/CIFAKE-image-dataset...")
    ds_train = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="train")
    print(f"[*] Total train available: {len(ds_train)} images")

    print("[*] Loading test split from dragonintelligence/CIFAKE-image-dataset...")
    ds_test = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="test")
    print(f"[*] Total test available: {len(ds_test)} images")

    # Filter by label
    train_real_indices = [i for i, label in enumerate(ds_train["label"]) if label == 0]
    train_fake_indices = [i for i, label in enumerate(ds_train["label"]) if label == 1]

    random.shuffle(train_real_indices)
    random.shuffle(train_fake_indices)

    n_train_per_class = max_train // 2
    n_val_per_class = max_val // 2

    selected_train_real = train_real_indices[:n_train_per_class]
    selected_train_fake = train_fake_indices[:n_train_per_class]

    selected_val_real = train_real_indices[n_train_per_class:n_train_per_class + n_val_per_class]
    selected_val_fake = train_fake_indices[n_train_per_class:n_train_per_class + n_val_per_class]

    # Test split
    test_real_indices = [i for i, label in enumerate(ds_test["label"]) if label == 0]
    test_fake_indices = [i for i, label in enumerate(ds_test["label"]) if label == 1]
    random.shuffle(test_real_indices)
    random.shuffle(test_fake_indices)

    n_test_per_class = max_test // 2
    selected_test_real = test_real_indices[:n_test_per_class]
    selected_test_fake = test_fake_indices[:n_test_per_class]

    def save_images(dataset, indices, target_dir_name, class_name):
        dest_dir_fake = os.path.join(output_dir, target_dir_name, "fake")
        dest_dir_ai = os.path.join(output_dir, target_dir_name, "ai_generated")
        dest_dir_real = os.path.join(output_dir, target_dir_name, "real")

        for idx, ds_idx in enumerate(tqdm(indices, desc=f"Saving {target_dir_name}/{class_name}")):
            sample = dataset[ds_idx]
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)

            fname = f"{class_name}_{idx:05d}.png"
            if class_name == "real":
                img.save(os.path.join(dest_dir_real, fname))
            else:
                img.save(os.path.join(dest_dir_fake, fname))
                # Also save in ai_generated for compatibility
                img.save(os.path.join(dest_dir_ai, fname))

    print(f"[*] Extracting {n_train_per_class * 2} training images...")
    save_images(ds_train, selected_train_real, "train", "real")
    save_images(ds_train, selected_train_fake, "train", "fake")

    print(f"[*] Extracting {n_val_per_class * 2} validation images...")
    save_images(ds_train, selected_val_real, "val", "real")
    save_images(ds_train, selected_val_fake, "val", "fake")

    print(f"[*] Extracting {n_test_per_class * 2} test images...")
    save_images(ds_test, selected_test_real, "test", "real")
    save_images(ds_test, selected_test_fake, "test", "fake")

    print("\n" + "=" * 60)
    print("DATASET EXTRACTION COMPLETE")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        r_count = len(os.listdir(os.path.join(output_dir, split, "real")))
        f_count = len(os.listdir(os.path.join(output_dir, split, "fake")))
        print(f" {split:6s} -> Real: {r_count:5d} | Fake: {f_count:5d}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CIFAKE dataset")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--max_train", type=int, default=8000, help="Total training images (balanced)")
    parser.add_argument("--max_val", type=int, default=1600, help="Total val images (balanced)")
    parser.add_argument("--max_test", type=int, default=1600, help="Total test images (balanced)")
    args = parser.parse_args()

    prepare_cifake(output_dir=args.output_dir, max_train=args.max_train, max_val=args.max_val, max_test=args.max_test)
