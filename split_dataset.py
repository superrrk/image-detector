import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = 'real_vs_fake'  # Where your original real/fake folders are
DEST_DIR = 'data'            # Where you want train/val to go
SPLIT_RATIO = 0.8            # 80% train, 20% val

def split_class(class_name):
    src_class_path = Path(SOURCE_DIR) / class_name
    all_images = list(src_class_path.glob('*.jpg'))  # Or *.png if needed
    random.shuffle(all_images)

    split_idx = int(len(all_images) * SPLIT_RATIO)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    for img in train_images:
        dest = Path(DEST_DIR) / 'train' / class_name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dest / img.name)

    for img in val_images:
        dest = Path(DEST_DIR) / 'val' / class_name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dest / img.name)

    print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")

if __name__ == "__main__":
    split_class('real')
    split_class('fake')
    print("✅ Dataset split complete.")
