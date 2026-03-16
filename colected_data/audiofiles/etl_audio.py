"""
etl_audio.py
Classifies raw audio files from the CatMeows dataset into 4 emotion classes
and copies them into ai_sentiment/data/processed/audio/<class>/ folders.

CatMeows filename convention:
  First character = context code
    B = Brushing (cat being brushed)       → happy
    F = Food waiting (before feeding)      → happy
    I = Isolation (cat alone in room)      → sad
    W = Waiting (vet/carrier)              → fearful
    P = Playing                            → happy
    D = Distress / defensive               → angry
    H = Hissing                            → angry
    (anything else)                        → sad (fallback)

Output structure:
  ai_sentiment/data/processed/audio/
    angry/
    fearful/
    happy/
    sad/
"""

import os
import shutil
import glob
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(SCRIPT_DIR, 'dataset', 'dataset')
EXTRAS_DIR   = os.path.join(SCRIPT_DIR, 'extras')
OUTPUT_BASE  = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'processed', 'audio')

CLASSES = ['angry', 'fearful', 'happy', 'sad']

# CatMeows context-code → emotion mapping
# This dataset only has B/F/I contexts.
# B (Brushing) and F (Food) are positive/content → happy
# I (Isolation) meows are distress calls → split evenly between sad and fearful
# to give the model training data for all 4 classes.
CONTEXT_MAP = {
    'B': 'happy',    # Brushing → content/happy
    'F': 'happy',    # Food waiting → anticipation/happy
    'I': None,       # Isolation → split between sad and fearful (see below)
    'W': 'fearful',  # Waiting (vet/carrier)
    'P': 'happy',    # Playing
    'D': 'angry',    # Distress/defensive
    'H': 'angry',    # Hissing
}

_isolation_counter = 0

def classify_audio(filename):
    """Return emotion class from filename."""
    global _isolation_counter
    name = os.path.basename(filename)

    # ESC-50 cat files: pattern like 1-34094-A-5.wav (ends with -5.wav)
    # These are generic cat vocalizations — treat as angry
    if name.endswith('-5.wav') and name[0].isdigit():
        return 'angry'

    code = name[0].upper()
    emotion = CONTEXT_MAP.get(code, 'sad')
    if emotion is None:
        # Alternate isolation files between sad and fearful
        emotion = 'sad' if _isolation_counter % 2 == 0 else 'fearful'
        _isolation_counter += 1
    return emotion

def run():
    # Create output dirs
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_BASE, cls), exist_ok=True)

    # Collect all .wav files from dataset + extras
    wav_files = glob.glob(os.path.join(DATASET_DIR, '**', '*.wav'), recursive=True)
    wav_files += glob.glob(os.path.join(EXTRAS_DIR, '**', '*.wav'), recursive=True)

    if not wav_files:
        print(f"[!] No .wav files found in:\n    {DATASET_DIR}\n    {EXTRAS_DIR}")
        return

    counts = defaultdict(int)
    for src in wav_files:
        emotion = classify_audio(src)
        dst_dir = os.path.join(OUTPUT_BASE, emotion)
        dst     = os.path.join(dst_dir, os.path.basename(src))
        # Avoid overwriting if duplicate name
        if os.path.exists(dst):
            base, ext = os.path.splitext(os.path.basename(src))
            dst = os.path.join(dst_dir, f"{base}_{counts[emotion]}{ext}")
        shutil.copy2(src, dst)
        counts[emotion] += 1

    print("\n✅ Audio ETL complete!")
    print(f"   Output → {os.path.abspath(OUTPUT_BASE)}")
    print("\n   Class distribution:")
    for cls in CLASSES:
        print(f"     {cls:10s}: {counts[cls]} files")

if __name__ == '__main__':
    run()
