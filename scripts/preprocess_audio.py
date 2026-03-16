"""
preprocess_audio.py
Pipeline to load audio files, convert to 16kHz WAV, remove silence,
reduce noise, extract MFCC features using librosa, and save features to processed folder.
"""

import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import noisereduce as nr
import soundfile as sf

# ── Config ─────────────────────────────────────────────────────────────────
# Get absolute path to the data inside the ai_sentiment folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "final_cleaned_audio")
TARGET_SR = 16000

def process_audio(file_path, output_class_dir):
    """Processes a single audio file and extracts/saves MFCCs and Spectrograms."""
    try:
        # Load audio, automatically converts to target sample rate (16kHz)
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        
        # Remove silence (trim leading/trailing silence below 20dB)
        y_trimmed, index = librosa.effects.trim(y, top_db=20)
        
        # Reduce noise
        y_denoised = nr.reduce_noise(y=y_trimmed, sr=sr)
        
        # Save processed wav
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        processed_wav_path = os.path.join(output_class_dir, f"{base_name}_cleaned.wav")
        sf.write(processed_wav_path, y_denoised, sr)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=40)
        
        # Save MFCC arrays as .npy files
        np.save(os.path.join(output_class_dir, f"{base_name}_mfcc.npy"), mfccs)
        
        return True
        
    except Exception as e:
        print(f"    [!] Error processing {os.path.basename(file_path)}: {e}")
        return False

def main():
    print(f"\n🎧 Audio Preprocessing Starting...")
    print(f"   Input: {INPUT_DIR}")
    print(f"   Output: {OUTPUT_DIR}\n")

    if not os.path.exists(INPUT_DIR):
        print(f"[!] Input directory {INPUT_DIR} not found.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    classes = ['angry', 'fearful', 'happy', 'sad']
    for cls in classes:
        cls_path = os.path.join(INPUT_DIR, cls)
        if not os.path.isdir(cls_path):
            print(f"  [!] Skipping {cls}, directory not found.")
            continue
            
        out_cls_path = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(out_cls_path, exist_ok=True)
        
        audio_paths = glob.glob(os.path.join(cls_path, '*.wav'))
        print(f"  [{cls}] Processing {len(audio_paths)} files...")
        
        count = 0
        for i, p in enumerate(audio_paths):
            if process_audio(p, out_cls_path):
                count += 1
            if (i + 1) % 50 == 0:
                print(f"    - Processed {i+1}/{len(audio_paths)} files...")
            
    print("\n✅ Audio preprocessing complete! Cleaned files saved to final_cleaned_audio.")

if __name__ == "__main__":
    main()
