import os
import hashlib
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("ADDING DIFFUSION-DB TO TRAINING DATASET")
    fake_dir = os.path.join("data", "train", "fake")
    os.makedirs(fake_dir, exist_ok=True)
    
    try:
        # Load the subset of DiffusionDB (fast streaming)
        dataset = load_dataset('poloclub/diffusiondb', '2m_first_1k', split='train', streaming=True, trust_remote_code=True)
        count = 0
        
        for item in tqdm(dataset, desc="Downloading DiffusionDB"):
            img = item.get('image')
            if img:
                img_hash = hashlib.md5(img.tobytes()).hexdigest()
                save_path = os.path.join(fake_dir, f"diffdb_{img_hash}.jpg")
                img.convert("RGB").save(save_path, "JPEG", quality=90)
                count += 1
                
        print(f"\n[OK] Added {count} new DiffusionDB (AI-Generated) images!")
    except Exception as e:
        print(f"Error fetching DiffusionDB: {e}")

if __name__ == '__main__':
    main()
