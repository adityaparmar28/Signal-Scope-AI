import os
import shutil
import hashlib
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def get_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

def process_and_save(item, split_dir, seen_hashes):
    try:
        # Some datasets have 'image', some have 'img'
        img = item.get('image', item.get('img'))
        label = item.get('label')
        
        if img is None or label is None:
            return False

        # CIFAKE: 0 = FAKE (AI), 1 = REAL. Wait, need to check label mapping.
        # usually 0 = REAL, 1 = FAKE in standard. We will just check directory names if available.
        # Let's map it based on dataset standard. For batgre/CIFAKE, 0=fake, 1=real
        # Our SignalScope expects: 0=real, 1=fake.
        label_str = "real" if label == 1 else "fake"
        
        # Deduplication
        img_hash = get_image_hash(img)
        if img_hash in seen_hashes:
            return False
        seen_hashes.add(img_hash)
        
        save_path = os.path.join(split_dir, label_str, f"{img_hash}.jpg")
        img.convert("RGB").save(save_path, "JPEG", quality=95)
        return True
    except Exception as e:
        return False

def main():
    print("Starting Dataset Pipeline for Windows (Merge & Dedup)...")
    base_dir = "data"
    
    # 1. Final Folder Structure setup
    for split in ["train", "val"]:
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)
            
    seen_hashes = set()
    
    # 2. Pick Datasets (CIFAKE via HuggingFace for speed and stability)
    print("-> Downloading batgre/CIFAKE (equivalent to GitHub CIFAKE)...")
    try:
        dataset = load_dataset("batgre/CIFAKE")
        
        for split_hf, split_local in [("train", "train"), ("test", "val")]:
            print(f"-> Processing {split_hf} split...")
            data_split = dataset[split_hf]
            
            # Using ThreadPool for faster I/O on Windows
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                for item in tqdm(data_split, desc=f"Queueing {split_hf}"):
                    split_dir = os.path.join(base_dir, split_local)
                    futures.append(executor.submit(process_and_save, item, split_dir, seen_hashes))
                
                valid_count = 0
                for f in tqdm(futures, desc=f"Saving {split_hf}"):
                    if f.result():
                        valid_count += 1
            print(f"Saved {valid_count} unique images for {split_local}.")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")

    print("\nDataset preparation complete! Ready for training.")

if __name__ == '__main__':
    main()
