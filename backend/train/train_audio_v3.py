import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import sys
import noisereduce as nr

# Configuration - HARDCODED ABSOLUTE PATHS
DATA_PATH = r"C:\projectmeww\ai_emotion\data\raw\audio"
MODEL_SAVE_PATH = r"C:\projectmeww\ai_emotion\backend\models\cat_emotion_lstm.keras"
CLASSES = ['angry', 'fearful', 'happy', 'sad']

def extract_audio_features_local(file_path):
    # Match utils.py logic exactly
    TARGET_AUDIO_SR = 16000
    MAX_PAD_LEN = 100
    
    y, sr = librosa.load(file_path, sr=TARGET_AUDIO_SR)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    y_denoised = nr.reduce_noise(y=y_trimmed, sr=sr)
    mfccs = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=40)
    
    if mfccs.shape[1] > MAX_PAD_LEN:
        mfccs = mfccs[:, :MAX_PAD_LEN]
    else:
        pad_width = MAX_PAD_LEN - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0), (0, pad_width)), mode='constant')
        
    features = np.transpose(mfccs)
    return np.expand_dims(features, axis=0)

def load_data():
    X = []
    y = []
    
    print(f"Data Path: {DATA_PATH}")
    for label in CLASSES:
        folder = os.path.join(DATA_PATH, label)
        if not os.path.exists(folder):
            print(f"FAILED: Folder {folder} not found.")
            continue
            
        files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.ogg'))]
        print(f"Label: {label}, Found: {len(files)} files")
        
        for file in files:
            path = os.path.join(folder, file)
            try:
                feat = extract_audio_features_local(path)
                X.append(feat[0])
                y.append(label)
            except Exception as e:
                print(f"Error {path}: {e}")
                    
    return np.array(X), np.array(y)

print("--- Audio Training (Self-Contained) ---")
X, y = load_data()

if len(X) == 0:
    print("Error: Total samples collected is 0.")
    sys.exit(1)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=len(CLASSES))

model = Sequential([
    Input(shape=(100, 40)),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    LSTM(32),
    Dropout(0.3),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(f"Training on {len(X)} samples for 50 epochs...")
model.fit(X, y_categorical, epochs=50, batch_size=4)

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Done! Saved to {MODEL_SAVE_PATH}")
