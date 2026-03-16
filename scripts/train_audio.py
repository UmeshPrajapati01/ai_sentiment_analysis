"""
train_audio.py
Trains an LSTM model on pre-extracted MFCC features for cat emotion recognition.
High-quality dataset: processed, balanced, and noise-reduced.
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "final_cleaned_audio")
MODEL_OUT = os.path.join(BASE_DIR, "models", "cat_emotion_lstm.keras")

CLASSES = ['angry', 'fearful', 'happy', 'sad']
MAX_LEN = 150  # Fixed time steps for MFCCs

def load_data():
    X = []
    y = []
    
    print("\n📦 Loading processed MFCC features...")
    
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        npy_files = glob.glob(os.path.join(cls_dir, '*_mfcc.npy'))
        
        print(f"  [{cls}] Found {len(npy_files)} samples.")
        
        for npy_path in npy_files:
            try:
                # Load MFCC (shape: n_mfcc, time)
                mfcc = np.load(npy_path)
                
                # Transpose to (time, n_mfcc)
                mfcc = mfcc.T
                
                # Zero-pad or truncate to fixed length
                if mfcc.shape[0] > MAX_LEN:
                    mfcc = mfcc[:MAX_LEN, :]
                else:
                    pad_width = MAX_LEN - mfcc.shape[0]
                    mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
                
                X.append(mfcc)
                y.append(cls)
            except Exception as e:
                print(f"    [!] Error loading {os.path.basename(npy_path)}: {e}")

    X = np.array(X)
    y = np.array(y)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(CLASSES))
    
    return X, y_categorical, le

def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def run():
    # 1. Load data
    X, y, le = load_data()
    
    if len(X) == 0:
        print("[!] No data loaded. Check paths.")
        return

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n📊 Data Summary:")
    print(f"   Total samples: {len(X)}")
    print(f"   Train size:    {len(X_train)}")
    print(f"   Test size:     {len(X_test)}")
    print(f"   Feature shape: {X.shape[1:]}")
    
    # 3. Build model
    model = build_model(input_shape=X.shape[1:], num_classes=len(CLASSES))
    model.summary()
    
    # 4. Train
    print("\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # 5. Save
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\n✅ Model saved to: {MODEL_OUT}")
    
    # Print final test accuracy
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"📈 Final Test Accuracy: {accuracy*100:.2f}%")

if __name__ == '__main__':
    run()
