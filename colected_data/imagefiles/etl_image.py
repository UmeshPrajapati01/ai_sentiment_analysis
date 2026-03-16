"""
etl_image.py
Reads YOLO-labelled images from train/valid/test splits, maps the 7 Roboflow
classes down to 4 model classes, and copies images into:
  ai_sentiment/data/processed/images/<class>/

Roboflow class indices (from data.yaml):
  0=angry  1=disgusted  2=happy  3=normal  4=sad  5=scared  6=surprised

Mapping to 4 model classes:
  angry     → angry
  disgusted → angry
  happy     → happy
  normal    → happy   (relaxed/content)
  sad       → sad
  scared    → fearful
  surprised → fearful
"""

import os
import shutil
import glob
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'processed', 'images')

SPLITS = ['train', 'valid', 'test']
CLASSES = ['angry', 'fearful', 'happy', 'sad']

# YOLO class index → model emotion
YOLO_TO_EMOTION = {
    0: 'angry',    # angry
    1: 'angry',    # disgusted → angry
    2: 'happy',    # happy
    3: 'happy',    # normal → happy (relaxed)
    4: 'sad',      # sad
    5: 'fearful',  # scared → fearful
    6: 'fearful',  # surprised → fearful
}

def get_class_from_label(label_path):
    """Read first YOLO label line and return class index."""
    try:
        with open(label_path, 'r') as f:
            line = f.readline().strip()
        if line:
            return int(line.split()[0])
    except Exception:
        pass
    return None

def run():
    # Create output dirs
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_BASE, cls), exist_ok=True)

    counts   = defaultdict(int)
    skipped  = 0

    for split in SPLITS:
        img_dir   = os.path.join(SCRIPT_DIR, split, 'images')
        label_dir = os.path.join(SCRIPT_DIR, split, 'labels')

        if not os.path.isdir(img_dir):
            print(f"[!] Skipping missing split: {split}")
            continue

        img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
        img_files += glob.glob(os.path.join(img_dir, '*.jpeg'))
        img_files += glob.glob(os.path.join(img_dir, '*.png'))

        for img_path in img_files:
            stem       = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, stem + '.txt')

            if not os.path.exists(label_path):
                # Fallback: try to infer from filename suffix
                emotion = infer_from_filename(stem)
                if emotion is None:
                    skipped += 1
                    continue
            else:
                cls_idx = get_class_from_label(label_path)
                if cls_idx is None:
                    skipped += 1
                    continue
                emotion = YOLO_TO_EMOTION.get(cls_idx)
                if emotion is None:
                    skipped += 1
                    continue

            dst_dir = os.path.join(OUTPUT_BASE, emotion)
            dst     = os.path.join(dst_dir, f"{split}_{os.path.basename(img_path)}")
            shutil.copy2(img_path, dst)
            counts[emotion] += 1

    print("\n✅ Image ETL complete!")
    print(f"   Output → {os.path.abspath(OUTPUT_BASE)}")
    print(f"   Skipped (no label): {skipped}")
    print("\n   Class distribution:")
    for cls in CLASSES:
        print(f"     {cls:10s}: {counts[cls]} images")

def infer_from_filename(stem):
    """
    Fallback: infer class from filename suffix pattern like:
    1_a_jpeg → angry, 1_h_jpeg → happy, etc.
    """
    name = stem.lower()
    # Check for known suffix patterns
    suffix_map = {
        '_a_': 'angry',
        '_d_': 'angry',    # disgusted
        '_h_': 'happy',
        '_n_': 'happy',    # normal
        '_s_': 'fearful',  # scared
        '_sad_': 'sad',
        '_surp_': 'fearful',  # surprised
    }
    for suffix, emotion in suffix_map.items():
        if suffix in name:
            return emotion
    return None

if __name__ == '__main__':
    run()
