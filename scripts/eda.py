"""
eda.py
Exploratory Data Analysis script to visualize class distribution for both images and datasets,
visualize spectrograms, and detect imbalanced classes.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import glob

# Paths
IMAGE_DIR = "../data/raw/images"
AUDIO_DIR = "../data/raw/audio"
EDA_OUTPUT_DIR = "../data/processed/eda"

def analyze_and_plot_distribution(data_dir, title, save_name):
    """
    Plots a bar chart showing class distribution and detects datset imbalance.
    """
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Skipping {title}.")
        return
        
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    counts = {}
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        # Count only files (images/audio)
        files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        counts[cls] = len(files)
            
    if all(count == 0 for count in counts.values()) or not counts:
        print(f"\nNo data found in {data_dir} to visualize.")
        return
        
    df = pd.DataFrame(list(counts.items()), columns=['Class', 'Count'])
    max_count = df['Count'].max()
    min_count = df['Count'].min()
    
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))
    
    # Imbalance detection logic
    if max_count == 0:
        pass
    elif max_count / max(min_count, 1) > 2.0:
        print("-> ⚠️ ALERT: Significant dataset imbalance detected (> 2:1 ratio). Consider adding augmented samples for minority classes.")
    else:
        print("-> ✅ Dataset appears reasonably balanced.")
        
    # Plotting
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Class', y='Count', data=df, palette='viridis')
    plt.title(title)
    plt.ylabel('Number of Samples')
    plt.xlabel('Emotion Category')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_OUTPUT_DIR, save_name))
    plt.close()
    print(f"Saved distribution plot to {save_name}")

def visualize_sample_spectrogram(audio_dir):
    """Visualizes a spectrogram from a random sample in the audio dataset."""
    if not os.path.exists(audio_dir):
        return
        
    audio_file = None
    classes = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
    for cls in classes:
        cls_path = os.path.join(audio_dir, cls)
        files = glob.glob(os.path.join(cls_path, '*.wav')) + glob.glob(os.path.join(cls_path, '*.mp3'))
        if files:
            audio_file = files[0]
            break
                
    if audio_file:
        print(f"\n=== Generating Sample Spectrogram ===")
        print(f"File Source: {audio_file}")
        
        y, sr = librosa.load(audio_file)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Sample Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_OUTPUT_DIR, 'sample_spectrogram_eda.png'))
        plt.close()
        print("Saved sample spectrogram EDA plot successfully.")

def main():
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
    
    print("Initiating EDA Pipeline...")
    analyze_and_plot_distribution(IMAGE_DIR, "Image Class Distribution", "image_distribution.png")
    analyze_and_plot_distribution(AUDIO_DIR, "Audio Class Distribution", "audio_distribution.png")
    visualize_sample_spectrogram(AUDIO_DIR)
    print("\nEDA Pipeline Complete. Results are available in data/processed/eda/")

if __name__ == "__main__":
    main()
