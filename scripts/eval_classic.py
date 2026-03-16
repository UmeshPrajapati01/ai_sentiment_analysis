import joblib
import os
import glob
import numpy as np
import librosa
from sklearn.metrics import accuracy_score, classification_report

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "audio")
MODEL_PATH = os.path.join(BASE_DIR, "models", "cat_emotion_rfc.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "audio_scaler_classic.pkl")
CLASSES = ['angry', 'fearful', 'happy', 'sad']

def extract_feat(f):
    y, sr = librosa.load(f, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(librosa.power_to_db(mel), axis=1)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    flat = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.concatenate([mfcc_mean, mfcc_std, chroma_mean, mel_mean, [np.mean(cent), np.mean(flat), np.mean(zcr)]])

if os.path.exists(MODEL_PATH):
    rfc = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Test on a few samples to see quality
    test_files = []
    test_labels = []
    for cls in CLASSES:
        fs = glob.glob(os.path.join(DATA_DIR, cls, '*.wav'))[:10] # 10 samples each
        test_files.extend(fs)
        test_labels.extend([cls]*len(fs))
    
    X_test = [extract_feat(f) for f in test_files]
    X_test = scaler.transform(X_test)
    preds = rfc.predict(X_test)
    
    le_path = os.path.join(BASE_DIR, "models", "audio_le.pkl")
    le = joblib.load(le_path)
    pred_labels = le.inverse_transform(preds)
    
    acc = accuracy_score(test_labels, pred_labels)
    print(f"Test Sample Accuracy: {acc*100:.2f}%")
    print(classification_report(test_labels, pred_labels))
else:
    print("Model not found!")
