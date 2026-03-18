
import os
import sys
import numpy as np
import pandas as pd
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# CONFIGURATION
# ==========================================

# Determine Project Root based on script location
# Script: backend/models_training/audio_model.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

# Input Directory (Train Data)
TRAIN_DATA_DIR = PROJECT_ROOT / "data_analysis/test_traindeddata/audio_data/train_data"

# Output Directory
OUTPUT_DIR = PROJECT_ROOT / "backend/trained_modelimages/audio_model"

# Audio Settings
SAMPLE_RATE = 22050
DURATION = 3 # Duration in seconds to process (fix length for consistency)

# Model Settings
N_ESTIMATORS = 100
RANDOM_STATE = 42

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(file_path):
    """
    Extracts audio features from a given file path.
    Features: MFCC, Chroma, Mel Spectrogram, Spectral Contrast.
    Returns a 1D numpy array of concatenated features.
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad audio if shorter than DURATION (to ensure consistent shape if needed, 
        # though we take mean/std so time dimension is collapsed)
        if len(audio) < SAMPLE_RATE * DURATION:
            padding = int(SAMPLE_RATE * DURATION) - len(audio)
            audio = np.pad(audio, (0, padding))
            
        # 1. MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        # 2. Chroma (Pitch)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # 3. Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)
        
        # 4. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        
        # 5. Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)
        
        # Concatenate all features
        features = np.hstack([mfcc_mean, chroma_mean, mel_mean, contrast_mean, tonnetz_mean])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# ==========================================
# DATA LOADING
# ==========================================

def load_data(data_dir):
    """
    Scans the directory for classes and files, extracts features.
    Returns: X (features), y (labels), classes (list of class names)
    """
    print(f"Scanning data from: {data_dir}")
    
    if not data_dir.exists():
        print(f"Error: Training directory not found at {data_dir}")
        sys.exit(1)
        
    classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    if not classes:
        print("Error: No class subdirectories found.")
        sys.exit(1)
        
    print(f"Detected {len(classes)} classes: {classes}")
    
    features_list = []
    labels_list = []
    
    for class_name in classes:
        class_dir = data_dir / class_name
        files = list(class_dir.glob("*.mp3")) + list(class_dir.glob("*.wav"))
        
        print(f"Processing class '{class_name}': {len(files)} files found.")
        
        for file_path in files:
            features = extract_features(file_path)
            if features is not None:
                features_list.append(features)
                labels_list.append(class_name)
                
    return np.array(features_list), np.array(labels_list), classes

# ==========================================
# VISUALIZATION
# ==========================================

def plot_confusion_matrix(y_true, y_pred, classes):
    """Generates and saves a confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved.")

def plot_class_distribution(y, classes):
    """Generates and saves a bar chart of class distribution."""
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(y, return_counts=True)
    # Ensure order matches classes list mapping if needed, generally useful to see raw counts
    sns.barplot(x=unique, y=counts, palette='viridis')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png")
    plt.close()
    print("Class distribution plot saved.")

# ==========================================
# TRAINING PIPELINE
# ==========================================

def train_pipeline():
    print("="*40)
    print("STARTING AUDIO TRAINING PIPELINE")
    print("="*40)
    
    # 1. Load Data
    X, y, classes = load_data(TRAIN_DATA_DIR)
    
    if len(X) == 0:
        print("Error: No data loaded.")
        return

    print(f"Data loaded. Features shape: {X.shape}, Labels shape: {y.shape}")
    
    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save Label Encoder
    joblib.dump(le, OUTPUT_DIR / "label_encoder.pkl")
    print(f"Label encoder saved to {OUTPUT_DIR / 'label_encoder.pkl'}")
    
    # 3. Scale Features (Important for many models)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save Scaler (Needed for inference)
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    print("Scaler saved.")
    
    # 4. Split Data (Train/Val split for internal validation)
    # We use the provided 'train_data' for training, but we split it here 
    # to evaluate the model performance before final deployment.
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    
    # 5. Train Model
    print("Training RandomForest Classifier...")
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1) # n_jobs=-1 uses all CPU cores
    model.fit(X_train, y_train)
    
    # 6. Evaluation
    print("Evaluating model...")
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Save Classification Report
    report = classification_report(y_val, y_pred, target_names=le.classes_)
    print("\nClassification Report:\n")
    print(report)
    
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(f"Validation Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
        
    # 7. Visualizations
    plot_confusion_matrix(y_val, y_pred, le.classes_)
    plot_class_distribution(y, le.classes_)
    
    # Save Feature Importance (Top 20)
    feature_importances = pd.Series(model.feature_importances_)
    plt.figure(figsize=(10,6))
    feature_importances.nlargest(20).plot(kind='barh')
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png")
    plt.close()
    
    # 8. Save Model
    joblib.dump(model, OUTPUT_DIR / "audio_classifier.pkl")
    print(f"Trained model saved to {OUTPUT_DIR / 'audio_classifier.pkl'}")
    
    print("="*40)
    print("TRAINING COMPLETE SUCCESSFULLY")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    train_pipeline()
