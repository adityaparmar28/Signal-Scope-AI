"""
FAST Multi-Generator Dataset Downloader for SignalScope.
Uses HuggingFace streaming to download only what we need — no full dataset caching.

Datasets:
1. GenImage_MidJourney (Midjourney)
2. GenImage_ADM (Ablated Diffusion Model) 
3. GenImage_BigGAN (BigGAN — GAN-based)
4. GenImage_glide (OpenAI GLIDE)

Each adds ~1000 fake + ~1000 real images → total ~8000+ diverse images combined with CIFAKE.
"""
import os
import sys
import random
from PIL import Image
from itertools import islice


def download_streaming(dataset_name, output_dir, max_fake=1000, max_real=1000, seed=42):
    """Download using streaming (fast, no full dataset cache)."""
    from datasets import load_dataset
    
    random.seed(seed)
    short_name = dataset_name.split("/")[-1].lower().replace("genimage_", "")
    
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset_name} (streaming)")
    print(f"  Target: {max_fake} fake + {max_real} real")
    print(f"{'='*60}")
    
    # Create dirs
    for split in ["train", "val"]:
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
    
    # Stream the dataset
    print(f"[*] Opening stream for {dataset_name}...")
    try:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    except Exception as e:
        print(f"[ERROR] Could not stream {dataset_name}: {e}")
        return
    
    real_count = 0
    fake_count = 0
    total_needed = max_fake + max_real
    
    # 80/20 train/val split
    val_every = 5  # every 5th image goes to val
    
    for sample in ds:
        label = sample.get("label", sample.get("labels", -1))
        img = sample["image"]
        
        if label == 0 and real_count < max_real:
            real_count += 1
            cls = "real"
            split = "val" if real_count % val_every == 0 else "train"
            idx = real_count
        elif label == 1 and fake_count < max_fake:
            fake_count += 1
            cls = "fake"
            split = "val" if fake_count % val_every == 0 else "train"
            idx = fake_count
        else:
            if real_count >= max_real and fake_count >= max_fake:
                break
            continue
        
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB").resize((224, 224), Image.LANCZOS)
        
        fname = f"{short_name}_{cls}_{idx:05d}.png"
        img.save(os.path.join(output_dir, split, cls, fname))
        
        total = real_count + fake_count
        if total % 100 == 0:
            print(f"  Progress: {total}/{total_needed} (real={real_count}, fake={fake_count})")
    
    print(f"[OK] {dataset_name}: saved {real_count} real + {fake_count} fake images")


def resize_cifake(data_dir):
    """Resize existing CIFAKE 32x32 images to 224x224."""
    print(f"\n{'='*60}")
    print("Resizing existing CIFAKE images from 32x32 -> 224x224")
    print(f"{'='*60}")
    
    count = 0
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            d = os.path.join(data_dir, split, cls)
            if not os.path.exists(d):
                continue
            for fname in os.listdir(d):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                # Skip already-resized multi-generator images  
                if any(prefix in fname for prefix in ["midjourney_", "adm_", "biggan_", "glide_"]):
                    continue
                fpath = os.path.join(d, fname)
                try:
                    img = Image.open(fpath)
                    w, h = img.size
                    if w < 224 or h < 224:
                        img = img.convert("RGB").resize((224, 224), Image.LANCZOS)
                        img.save(fpath)
                        count += 1
                except Exception:
                    pass
    
    print(f"[OK] Resized {count} CIFAKE images to 224x224")


def print_summary(output_dir):
    print(f"\n{'='*60}")
    print("FINAL MULTI-GENERATOR DATASET SUMMARY")
    print(f"{'='*60}")
    total = 0
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            p = os.path.join(output_dir, split, cls)
            if os.path.exists(p):
                count = len([f for f in os.listdir(p) if f.endswith((".png", ".jpg", ".jpeg"))])
                print(f"  {split:6s}/{cls:5s}: {count:6d} images")
                total += count
    print(f"  {'TOTAL':12s}: {total:6d} images")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--max_per_gen", type=int, default=1000)
    parser.add_argument("--skip_resize", action="store_true")
    args = parser.parse_args()

    N = args.max_per_gen
    
    # Step 0: Resize existing CIFAKE from 32x32 to 224x224
    if not args.skip_resize:
        resize_cifake(args.output_dir)
    
    # Step 1-4: Download from 4 diverse generators
    datasets = [
        "bitmind/GenImage_MidJourney",   # Midjourney 
        "bitmind/GenImage_ADM",          # Ablated Diffusion Model
        "bitmind/GenImage_BigGAN",       # BigGAN (GAN, not diffusion)
        "bitmind/GenImage_glide",        # OpenAI GLIDE
    ]
    
    for ds_name in datasets:
        try:
            download_streaming(ds_name, args.output_dir, max_fake=N, max_real=N)
        except Exception as e:
            print(f"[ERROR] Failed {ds_name}: {e}")
            continue
    
    print_summary(args.output_dir)
    print("\n[NEXT] Retrain with:")
    print("  python src/train.py --epochs 10 --batch_size 32 --lr 3e-5")
