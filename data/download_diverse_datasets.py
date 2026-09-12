"""
Fast, direct downloader for Midjourney and Diverse Real/AI datasets.
Uses direct multithreaded HTTPS requests (no HuggingFace datasets library bottleneck).

Downloads:
1. Midjourney (ehristoforu/midjourney-images) -> fake/midjourney_*.jpg
2. Diverse AI Art (Hemg/AI-Generated-vs-Real-Images-Datasets) -> fake/aiart_*.jpg
3. Real Photos (Hemg/RealArt) -> real/realart_*.jpg
"""
import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def get_tree(repo_id, folder=""):
    url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main/{folder}".rstrip("/")
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code == 200:
        return [x["path"] for x in r.json()]
    return []


def download_image(args):
    repo_id, path, dest_path = args
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            # Validate image and resize to 224x224
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img = img.resize((224, 224), Image.LANCZOS)
            img.save(dest_path, "JPEG", quality=90)
            return True
    except Exception:
        pass
    return False


def fetch_dataset(repo_id, source_folder, target_cls, prefix, max_count=500, split_val_ratio=0.2):
    print(f"\n[*] Fetching list from {repo_id}/{source_folder}...")
    paths = get_tree(repo_id, source_folder)
    valid_paths = [p for p in paths if not p.endswith("_thumb.jpg") and p.lower().endswith((".jpg", ".png", ".webp"))]
    valid_paths = valid_paths[:max_count]
    print(f"    Found {len(valid_paths)} candidate images. Preparing download...")

    train_dir = os.path.join("data", "train", target_cls)
    val_dir = os.path.join("data", "val", target_cls)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    val_count = int(len(valid_paths) * split_val_ratio)
    train_paths = valid_paths[val_count:]
    val_paths = valid_paths[:val_count]

    tasks = []
    for i, p in enumerate(train_paths):
        dest = os.path.join(train_dir, f"{prefix}_train_{i:04d}.jpg")
        tasks.append((repo_id, p, dest))
    for i, p in enumerate(val_paths):
        dest = os.path.join(val_dir, f"{prefix}_val_{i:04d}.jpg")
        tasks.append((repo_id, p, dest))

    print(f"    Downloading {len(tasks)} images ({len(train_paths)} train, {len(val_paths)} val) with 12 workers...")
    success = 0
    with ThreadPoolExecutor(max_workers=12) as executor:
        for ok in executor.map(download_image, tasks):
            if ok:
                success += 1

    print(f"[OK] {prefix}: Successfully downloaded {success}/{len(tasks)} images to data/{target_cls}!")


def main():
    print("=" * 65)
    print("DOWNLOADING DIVERSE GENERATOR DATASETS (Midjourney + AI Art + Real)")
    print("=" * 65)

    # 1. Midjourney images -> fake
    fetch_dataset("ehristoforu/midjourney-images", "images", "fake", "midjourney", max_count=500)

    # 2. Diverse AI Generated Art -> fake
    fetch_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", "AiArtData/AiArtData", "fake", "aiart", max_count=450)

    # 3. Diverse Real Art & Photos -> real
    fetch_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", "RealArt/RealArt", "real", "realart", max_count=430)

    print("\n" + "=" * 65)
    print("DIVERSE DATASET INGESTION COMPLETE")
    print("=" * 65)
    for split in ["train", "val"]:
        r_cnt = len(os.listdir(os.path.join("data", split, "real")))
        f_cnt = len(os.listdir(os.path.join("data", split, "fake")))
        print(f"  data/{split}/: Real = {r_cnt:5d} | Fake = {f_cnt:5d}")


if __name__ == "__main__":
    main()
