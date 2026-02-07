# ...existing code...
"""
Audio cleaning, feature extraction and EDA utilities (B1-B4).

Functions:
- acquire_audio_files(input_dir): collect (path, label) pairs
- load_audio(path, sr=None): load audio with librosa
- remove_silence(y, sr): trim leading/trailing silence
- reduce_noise(y, sr): simple spectral-gating noise reduction
- extract_features(y, sr): MFCCs, RMS, centroid, zcr, pitch summary
- plot_pitch_intensity(y, sr, out_path): save pitch vs intensity plot
- process_dataset(input_dir, output_dir): run whole pipeline and save CSV + plots
"""
import os
import csv
from glob import glob
from typing import List, Tuple
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import signal

# A1: Acquire audio files organized by class subfolders
def acquire_audio_files(input_dir: str) -> List[Tuple[str, str]]:
    files = []
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    for label in sorted(os.listdir(input_dir)):
        label_dir = os.path.join(input_dir, label)
        if os.path.isdir(label_dir):
            for ext in ('*.wav', '*.mp3', '*.flac', '*.ogg'):
                for p in glob(os.path.join(label_dir, ext)):
                    files.append((p, label))
    return files

# A1 helper: load audio
def load_audio(path: str, sr: int = 22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

# B3: Remove silence (leading/trailing)
def remove_silence(y: np.ndarray, sr: int, top_db: int = 30):
    yt, index = librosa.effects.trim(y, top_db=top_db)
    return yt

# B3: Simple noise reduction via spectral gating (median noise estimate)
def reduce_noise(y: np.ndarray, sr: int, prop_decrease: float = 1.0, n_fft: int = 2048, hop_length: int = 512):
    S_full = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S_full), np.angle(S_full)
    # estimate noise from first 0.5s or first 10 frames
    noise_frames = max(1, int(0.5 * sr / hop_length))
    noise_mag = np.median(mag[:, :noise_frames], axis=1, keepdims=True)
    # create mask
    gain = np.maximum(0.0, (mag - prop_decrease * noise_mag)) / (mag + 1e-8)
    S_clean = gain * mag * np.exp(1j * phase)
    y_clean = librosa.istft(S_clean, hop_length=hop_length)
    return y_clean

# B2: Feature extraction (MFCCs, RMS, centroid, zcr, pitch summary)
def extract_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> dict:
    feats = {}
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    for i in range(n_mfcc):
        feats[f"mfcc{i+1}_mean"] = float(mfcc_mean[i])
        feats[f"mfcc{i+1}_std"] = float(mfcc_std[i])
    # RMS
    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["centroid_mean"] = float(np.mean(centroid))
    feats["centroid_std"] = float(np.std(centroid))
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    # Pitch estimation using piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = []
    for i in range(pitches.shape[1]):
        idx = magnitudes[:, i].argmax()
        p = pitches[idx, i]
        if p > 0:
            pitch_vals.append(p)
    if pitch_vals:
        feats["pitch_median"] = float(np.median(pitch_vals))
        feats["pitch_mean"] = float(np.mean(pitch_vals))
        feats["pitch_std"] = float(np.std(pitch_vals))
    else:
        feats["pitch_median"] = 0.0
        feats["pitch_mean"] = 0.0
        feats["pitch_std"] = 0.0
    return feats

# B4: Plot pitch vs intensity (RMS) and save one graph per audio file
def plot_pitch_intensity(y: np.ndarray, sr: int, out_path: str, title: str = ""):
    hop_length = 512
    # RMS
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    # Pitch via piptrack
    pitches, mags = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
    pitch_track = []
    for i in range(pitches.shape[1]):
        idx = mags[:, i].argmax()
        p = pitches[idx, i]
        pitch_track.append(p if p > 0 else np.nan)
    pitch_track = np.array(pitch_track)
    pitch_times = librosa.frames_to_time(np.arange(len(pitch_track)), sr=sr, hop_length=hop_length)
    # Plot
    plt.figure(figsize=(8,3.5))
    ax1 = plt.gca()
    ax1.plot(times, rms, color='C0', label='RMS Energy')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("RMS", color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax2 = ax1.twinx()
    ax2.plot(pitch_times, pitch_track, color='C1', label='Pitch (Hz)', alpha=0.8)
    ax2.set_ylabel("Pitch (Hz)", color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')
    plt.title(title if title else os.path.basename(out_path))
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

# Utility: write CSV header and rows
def write_features_csv(csv_path: str, rows: List[dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

# Process entire dataset
def process_dataset(input_dir: str, output_dir: str, denoise: bool = True, trim_db: int = 30):
    files = acquire_audio_files(input_dir)
    if not files:
        print("No audio files found in", input_dir); return
    features_rows = []
    # determine CSV fields using extract_features template
    sample_y, sample_sr = load_audio(files[0][0])
    sample_feats = extract_features(sample_y, sample_sr)
    fieldnames = ["filename", "label", "duration"] + list(sample_feats.keys())
    plots_dir = os.path.join(output_dir, "plots")
    csv_path = os.path.join(output_dir, "features.csv")
    for path, label in files:
        try:
            y, sr = load_audio(path)
            duration = librosa.get_duration(y=y, sr=sr)
            y = remove_silence(y, sr, top_db=trim_db)
            if denoise:
                y = reduce_noise(y, sr)
            feats = extract_features(y, sr)
            row = {"filename": os.path.basename(path), "label": label, "duration": duration}
            row.update(feats)
            features_rows.append(row)
            # save plot (one graph per file)
            plot_out = os.path.join(plots_dir, label, os.path.splitext(os.path.basename(path))[0] + "_pitch_rms.png")
            plot_pitch_intensity(y, sr, plot_out, title=f"{label} - {os.path.basename(path)}")
            print(f"Processed {path} -> label={label}")
        except Exception as e:
            print(f"Error {path}: {e}")
    # write CSV
    write_features_csv(csv_path, features_rows, fieldnames)
    print("Features saved to", csv_path)
    print("Plots saved to", plots_dir)

# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audio cleaning, feature extraction and EDA (B1-B4)")
    parser.add_argument("--input-dir", type=str,
                        default=r"C:\Users\ramak\OneDrive\Desktop\project_smartx\project_mew\Dev-of-an-AI-Based-Cat-Emotion-Recognition-System-Using-Facial-and-Vocal-Analysis_Feb_Batch-8_2026\uncleaned_datasets\dataset\classified_audio",
                        help="Root folder with class subfolders containing audio")
    parser.add_argument("--output-dir", type=str,
                        default=r"C:\Users\ramak\OneDrive\Desktop\project_smartx\project_mew\Dev-of-an-AI-Based-Cat-Emotion-Recognition-System-Using-Facial-and-Vocal-Analysis_Feb_Batch-8_2026\output_weekwise\week1output\audio",
                        help="Directory to save features and plots")
    parser.add_argument("--no-denoise", action="store_true", help="Disable denoising step")
    parser.add_argument("--trim-db", type=int, default=30, help="Silence trim top_db for librosa.effects.trim")
    args = parser.parse_args()
    process_dataset(args.input_dir, args.output_dir, denoise=not args.no_denoise, trim_db=args.trim_db)