"""
train_audio_sota_v3_classic.py
Using Classical ML (Random Forest/XGBoost) on Comprehensive Audio Features.
Often more robust than Deep Learning for datasets < 1000.
Target: 85-90%+ accuracy.
"""

import os
import glob
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "audio")
MODEL_OUT = os.path.join(BASE_DIR, "models", "cat_emotion_rfc.pkl")
SCALER_OUT = os.path.join(BASE_DIR, "models", "audio_scaler_classic.pkl")

CLASSES = ['angry', 'fearful', 'happy', 'sad']

def extract_comprehensive_features(file_path):
    """Extracts a wide range of statistical audio features."""
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # 1. MFCC (40)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # 2. Chroma (12)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # 3. Mel (128 -> Mean)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(librosa.power_to_db(mel), axis=1)
        
        # 4. Spectral Features
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        flat = librosa.feature.spectral_flatness(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Combine all into one vector
        features = np.concatenate([
            mfcc_mean, mfcc_std, 
            chroma_mean, 
            mel_mean, 
            [np.mean(cent), np.mean(flat), np.mean(zcr)]
        ])
        return features
    except:
        return None

def run():
    X, y = [], []
    print("\n🔍 Extracting Comprehensive Audio Features (Classic ML approach)...")
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        files = glob.glob(os.path.join(cls_dir, '*.wav'))
        print(f"  [{cls}] {len(files)} files")
        for f in files:
            feat = extract_comprehensive_features(f)
            if feat is not None:
                X.append(feat)
                y.append(cls)
                
    X = np.array(X)
    y = np.array(y)
    
    # ── Preprocessing ───────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # ── Training (Random Forest with Tuning) ─────────────────────
    print("\n🚀 Training Random Forest Classifier...")
    # RFC is very hard to overfit and handles small data well
    rfc = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
    rfc.fit(X_train, y_train)
    
    # ── Evaluation ──────────────────────────────────────────────
    accuracy = rfc.score(X_test, y_test)
    print(f"\n📈 Final Accuracy (Classic ML): {accuracy*100:.2f}%")
    
    # Save
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(rfc, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(le, os.path.join(os.path.dirname(MODEL_OUT), "audio_le.pkl"))
    
    print(f"✅ Classic Model saved to: {MODEL_OUT}")

if __name__ == '__main__':
    run()
