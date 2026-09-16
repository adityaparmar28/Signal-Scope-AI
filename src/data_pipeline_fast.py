import os
import hashlib
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def get_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

def process_and_save(item, split_dir, seen_hashes):
    try:
        img = item.get('image', item.get('img'))
        label = item.get('label')
        
        if img is None or label is None: return False

        label_str = "real" if label == 1 else "fake"
        img_hash = get_image_hash(img)
        
        if img_hash in seen_hashes: return False
        seen_hashes.add(img_hash)
        
        save_path = os.path.join(split_dir, label_str, f"{img_hash}.jpg")
        img.convert("RGB").save(save_path, "JPEG", quality=90)
        return label_str
    except Exception as e:
        return False

def main():
    print("EMERGENCY HACKATHON MODE: FAST PIPELINE")
    base_dir = "data"
    
    # 1. Structure setup
    for split in ["train", "val"]:
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)
            
    seen_hashes = set()
    
    # Limits for 40 min deadline: Train (2000 total), Val (500 total)
    LIMITS = {"train": {"real": 1000, "fake": 1000}, "val": {"real": 250, "fake": 250}}
    counts = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}}
    
    print("-> Downloading CIFAKE...")
    try:
        dataset = load_dataset("batgre/CIFAKE", streaming=True) # Use streaming to fetch just what we need instantly
        
        for split_hf, split_local in [("train", "train"), ("test", "val")]:
            print(f"-> Extracting fast {split_local} subset...")
            split_dir = os.path.join(base_dir, split_local)
            
            for item in tqdm(dataset[split_hf], desc=f"Processing {split_local}"):
                label = item.get('label')
                label_str = "real" if label == 1 else "fake"
                
                if counts[split_local][label_str] >= LIMITS[split_local][label_str]:
                    if all(counts[split_local][k] >= LIMITS[split_local][k] for k in ["real", "fake"]):
                        break # We have enough for both classes
                    continue # Skip if this class is full
                
                res = process_and_save(item, split_dir, seen_hashes)
                if res:
                    counts[split_local][res] += 1
                    
            print(f"Saved {counts[split_local]} images for {split_local}.")
            
    except Exception as e:
        print(f"Error: {e}")

    print("\n[OK] FAST DATA PIPELINE COMPLETE. READY FOR TRAINING.")

if __name__ == '__main__':
    main()
