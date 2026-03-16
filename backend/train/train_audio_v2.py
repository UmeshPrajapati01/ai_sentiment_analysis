import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import sys

# Add parent dir to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import extract_audio_features

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/raw/audio"))
MODEL_SAVE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../models/cat_emotion_lstm.keras"))
CLASSES = ['angry', 'fearful', 'happy', 'sad']


def load_data():
    X = []
    y = []
    
    print(f"Checking DATA_PATH: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"CRITICAL: DATA_PATH does not exist: {DATA_PATH}")
        return np.array(X), np.array(y)

    for label in CLASSES:
        folder = os.path.join(DATA_PATH, label)
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} not found.")
            continue
            
        print(f"Processing {label} in {folder}...")
        files = [f for f in os.listdir(folder) if f.endswith(('.wav', '.mp3', '.ogg'))]
        print(f"Found {len(files)} files.")
        for file in files:
            path = os.path.join(folder, file)
            try:
                feat = extract_audio_features(path)
                X.append(feat[0]) 
                y.append(label)
            except Exception as e:
                print(f"Error processing {path}: {e}")

                    
    return np.array(X), np.array(y)

print("--- Starting Audio Model Training ---")
X, y = load_data()

if len(X) == 0:
    print("Error: No data found. Check your paths.")
    sys.exit(1)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=len(CLASSES))

# Build Model
model = Sequential([
    Input(shape=(100, 40)),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    LSTM(32),
    Dropout(0.3),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"Training on {len(X)} samples...")
model.fit(X, y_categorical, epochs=50, batch_size=4, verbose=1)

# Ensure models directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

model.save(MODEL_SAVE_PATH)
print(f"Audio model saved to {MODEL_SAVE_PATH}")
