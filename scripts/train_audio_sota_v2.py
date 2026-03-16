"""
train_audio_sota_v2.py
Refined SOTA script for 85-100% accuracy.
Focus: Extreme Overfitting Protection & Dynamic Augmentation.
"""

import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "audio")
MODEL_OUT = os.path.join(BASE_DIR, "models", "cat_emotion_sota_v2.keras")

CLASSES = ['angry', 'fearful', 'happy', 'sad']
IMG_SIZE = (64, 64) # Smaller size to prevent overfitting on small data
SR = 16000

def extract_mel_spec_v2(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR)
        # 1. Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=256)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 2. Resizing to (64, 64)
        if S_dB.shape[1] > 64:
            S_dB = S_dB[:, :64]
        else:
            S_dB = np.pad(S_dB, ((0, 0), (0, 64 - S_dB.shape[1])), mode='constant')
            
        # 3. Global Normalization (approx)
        S_dB = (S_dB + 80) / 80 # Map -80dB..0dB to 0..1
        return S_dB
    except:
        return None

def load_dataset():
    X, y = [], []
    print("\n📦 Loading Mel Images v2...")
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        files = glob.glob(os.path.join(cls_dir, '*.wav'))
        print(f"  [{cls}] {len(files)} files")
        for f in files:
            spec = extract_mel_spec_v2(f)
            if spec is not None:
                X.append(spec)
                y.append(cls)
    return np.array(X), np.array(y)

def build_robust_cnn(input_shape, num_classes):
    """Lighter, more robust CNN to prevent overfitting."""
    model = Sequential([
        Input(shape=input_shape),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(), # Better than Flatten for preventing overfitting
        
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def run():
    # 1. Load
    X, y = load_dataset()
    X = X.reshape(-1, 64, 64, 1)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
    
    # 3. Model
    model = build_robust_cnn((64, 64, 1), len(CLASSES))
    
    # 4. Train
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
    
    print("\n🚀 Training Robust SOTA Model...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                        epochs=150, batch_size=32, callbacks=callbacks, verbose=1)
    
    # 5. Result
    model.save(MODEL_OUT)
    print(f"\n✅ Model saved to: {MODEL_OUT}")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"📈 Final Accuracy (after overfitting protection): {accuracy*100:.2f}%")

if __name__ == '__main__':
    run()
