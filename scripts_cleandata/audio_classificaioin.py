"""
Audio Classification Script

Use Case:
---------
This script automates the classification of unclassified cat audio files using machine learning.
It extracts MFCC features from audio files, trains a RandomForest classifier on the classified files,
and predicts the class for each unclassified file.

Requirements:
-------------
pip install librosa scikit-learn numpy

Usage:
------
python scripts_cleandata/audio_classificaioin.py
"""

import os
import shutil
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFIED_DIR = os.path.join(BASE_DIR, 'uncleaned_datasets', 'dataset', 'classified_audio')
UNCLASSIFIED_DIR = os.path.join(BASE_DIR, 'uncleaned_datasets', 'dataset', 'non_classification_audio')
OUTPUT_DIR = os.path.join(BASE_DIR, 'uncleaned_datasets', 'cleaned_data', 'classified_audiofiles')

def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def get_classes(classified_dir):
    """Get list of class names from classified audio directory."""
    return [d for d in os.listdir(classified_dir) if os.path.isdir(os.path.join(classified_dir, d))]

def load_classified_data(classified_dir, classes):
    """Load features and labels from classified audio files."""
    X, y = [], []
    for cls in classes:
        class_dir = os.path.join(classified_dir, cls)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                fpath = os.path.join(class_dir, fname)
                try:
                    features = extract_features(fpath)
                    X.append(features)
                    y.append(cls)
                except Exception as e:
                    print(f"Error processing {fpath}: {e}")
    return np.array(X), np.array(y)

def main():
    classes = get_classes(CLASSIFIED_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    print("Extracting features and training classifier...")
    X_train, y_train = load_classified_data(CLASSIFIED_DIR, classes)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    print("Classifying unclassified audio files...")
    for fname in os.listdir(UNCLASSIFIED_DIR):
        if fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            src_path = os.path.join(UNCLASSIFIED_DIR, fname)
            try:
                features = extract_features(src_path)
                predicted_class = clf.predict([features])[0]
                dst_path = os.path.join(OUTPUT_DIR, predicted_class, fname)
                shutil.copy2(src_path, dst_path)
                print(f"Classified {fname} as {predicted_class}")
            except Exception as e:
                print(f"Error processing {src_path}: {e}")

if __name__ == "__main__":
    main()