"""
augment_audio.py
Safely augments underrepresented audio classes (angry, sad) to balance
the dataset. Each original file gets max 5 augmented variants using
conservative transforms that preserve the emotional character of the audio.

Techniques used (all label-preserving):
  1. Pitch shift    (+/- 1-2 semitones)
  2. Time stretch   (0.85x or 1.15x speed)
  3. Add white noise (very low SNR ~30dB)
  4. Volume scale   (0.7x or 1.3x)
  5. Combined       (pitch + noise together)

Target: bring angry and sad up to ~240 files each.
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────
AUDIO_BASE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'data', 'processed', 'audio')
TARGET_COUNT = 1000   # Massive target to hit 99% accuracy
SR           = 16000 # resample rate

# Classes to augment (all to hit the 1000 target)
AUGMENT_CLASSES = ['angry', 'sad', 'fearful', 'happy']

# ── Augmentation functions ──────────────────────────────────────────────────

def pitch_shift(y, sr, steps):
    """Shift pitch by `steps` semitones."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def time_stretch(y, rate):
    """Stretch/compress time by `rate` (< 1 = slower, > 1 = faster)."""
    return librosa.effects.time_stretch(y, rate=rate)

def add_noise(y, snr_db=30):
    """Add white noise at given SNR in dB."""
    signal_power = np.mean(y ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), len(y))
    return (y + noise).astype(np.float32)

def volume_scale(y, factor):
    """Scale volume by factor, clip to [-1, 1]."""
    return np.clip(y * factor, -1.0, 1.0).astype(np.float32)

def combined(y, sr):
    """Pitch shift + light noise together."""
    y = pitch_shift(y, sr, steps=1.5)
    y = add_noise(y, snr_db=32)
    return y

# Augmentation pipeline — 5 variants per file
AUGMENTATIONS = [
    ('ps_up',      lambda y, sr: pitch_shift(y, sr, steps=1.5)),
    ('ps_down',    lambda y, sr: pitch_shift(y, sr, steps=-1.5)),
    ('ts_slow',    lambda y, sr: time_stretch(y, rate=0.88)),
    ('ts_fast',    lambda y, sr: time_stretch(y, rate=1.12)),
    ('vol_noise',  lambda y, sr: combined(y, sr)),
]

# ── Main ────────────────────────────────────────────────────────────────────

def augment_class(cls):
    cls_dir   = os.path.join(AUDIO_BASE, cls)
    orig_files = glob.glob(os.path.join(cls_dir, '*.wav'))
    orig_count = len(orig_files)

    if orig_count == 0:
        print(f"  [!] No files found in {cls}, skipping.")
        return 0

    needed    = TARGET_COUNT - orig_count
    if needed <= 0:
        print(f"  [{cls}] Already at {orig_count} files, no augmentation needed.")
        return 0

    print(f"  [{cls}] {orig_count} original → need {needed} more to reach {TARGET_COUNT}")

    generated = 0
    aug_idx   = 0

    # Cycle through original files and augmentations until we hit target
    while generated < needed:
        src_path = orig_files[aug_idx % orig_count]
        aug_name, aug_fn = AUGMENTATIONS[aug_idx % len(AUGMENTATIONS)]

        try:
            y, sr = librosa.load(src_path, sr=SR, res_type='kaiser_fast')
            y_aug = aug_fn(y, sr)

            # Ensure float32 and valid range
            y_aug = np.clip(y_aug, -1.0, 1.0).astype(np.float32)

            stem    = os.path.splitext(os.path.basename(src_path))[0]
            out_name = f"{stem}_aug_{aug_name}_{generated}.wav"
            out_path = os.path.join(cls_dir, out_name)

            sf.write(out_path, y_aug, SR, subtype='PCM_16')
            generated += 1
            
            if generated % 10 == 0:
                print(f"    - [{cls}] Generated {generated}/{needed} files...")

        except Exception as e:
            print(f"    [!] Failed on {os.path.basename(src_path)}: {e}")

        aug_idx += 1

    return generated

def run():
    print("\n🎵 Audio Augmentation Starting...\n")
    total = 0
    for cls in AUGMENT_CLASSES:
        n = augment_class(cls)
        total += n

    print(f"\n✅ Augmentation complete! {total} new files generated.")
    print("\n   Final class distribution:")
    for cls in ['angry', 'fearful', 'happy', 'sad']:
        cls_dir = os.path.join(AUDIO_BASE, cls)
        count   = len(glob.glob(os.path.join(cls_dir, '*.wav')))
        print(f"     {cls:10s}: {count} files")

if __name__ == '__main__':
    run()
