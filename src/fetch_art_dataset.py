import os
import sys
from PIL import Image
from datasets import load_dataset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_FAKE = os.path.join(BASE_DIR, 'data', 'train', 'fake')
TRAIN_REAL = os.path.join(BASE_DIR, 'data', 'train', 'real')
VAL_FAKE = os.path.join(BASE_DIR, 'data', 'val', 'fake')
VAL_REAL = os.path.join(BASE_DIR, 'data', 'val', 'real')

for d in [TRAIN_FAKE, TRAIN_REAL, VAL_FAKE, VAL_REAL]:
    os.makedirs(d, exist_ok=True)

print("[*] Streaming 'Hemg/AI-Generated-vs-Real-Images-Datasets' (Graphics, Paint, AI Art)...")
ds = load_dataset('Hemg/AI-Generated-vs-Real-Images-Datasets', split='train', streaming=True)

ai_count = 0
real_count = 0
TARGET_TRAIN = 200
TARGET_VAL = 50

for item in ds:
    img = item['image'].convert('RGB')
    label = item['label'] # 0 = AiArtData (AI), 1 = RealArt (Real)
    
    if label == 0 and ai_count < (TARGET_TRAIN + TARGET_VAL):
        if ai_count < TARGET_TRAIN:
            save_path = os.path.join(TRAIN_FAKE, f"art_ai_{ai_count:04d}.jpg")
        else:
            save_path = os.path.join(VAL_FAKE, f"art_ai_val_{ai_count - TARGET_TRAIN:04d}.jpg")
        img.resize((224, 224), Image.Resampling.BILINEAR).save(save_path, "JPEG", quality=90)
        ai_count += 1
        if ai_count % 100 == 0:
            print(f" [+] Collected {ai_count}/{TARGET_TRAIN + TARGET_VAL} AI Art images...")
            
    elif label == 1 and real_count < (TARGET_TRAIN + TARGET_VAL):
        if real_count < TARGET_TRAIN:
            save_path = os.path.join(TRAIN_REAL, f"art_real_{real_count:04d}.jpg")
        else:
            save_path = os.path.join(VAL_REAL, f"art_real_val_{real_count - TARGET_TRAIN:04d}.jpg")
        img.resize((224, 224), Image.Resampling.BILINEAR).save(save_path, "JPEG", quality=90)
        real_count += 1
        if real_count % 100 == 0:
            print(f" [+] Collected {real_count}/{TARGET_TRAIN + TARGET_VAL} Real Art images...")
            
    if ai_count >= (TARGET_TRAIN + TARGET_VAL) and real_count >= (TARGET_TRAIN + TARGET_VAL):
        break

print(f"\n[✓] Successfully added {ai_count} AI Art images and {real_count} Real Art images into training pipeline!")
