"""
train_audio_v2.py
Advanced Training Script to reach 75-90% accuracy.
Improvements:
1. Hybrid CNN-LSTM Architecture (Best for audio)
2. Delta and Delta-Delta features (Contextual dynamics)
3. Feature Normalization (StandardScaler)
4. Learning Rate Scheduler (Fine-tuning)
5. Increased complexity and Dropout control
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Bidirectional, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib # to save the scaler

# ── Config ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "final_cleaned_audio")
MODEL_OUT = os.path.join(BASE_DIR, "models", "cat_emotion_lstm_v2.keras")
SCALER_OUT = os.path.join(BASE_DIR, "models", "audio_scaler.pkl")

CLASSES = ['angry', 'fearful', 'happy', 'sad']
MAX_LEN = 150  
N_MFCC = 40

def add_deltas(mfcc):
    """Adds Delta and Delta-Delta features to capture movement in audio."""
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, delta, delta2], axis=0)

import librosa # needed for delta

def load_data():
    X = []
    y = []
    
    print("\n📦 Loading processed MFCC features and calculating Deltas...")
    
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        npy_files = glob.glob(os.path.join(cls_dir, '*_mfcc.npy'))
        
        print(f"  [{cls}] Found {len(npy_files)} samples.")
        
        for npy_path in npy_files:
            try:
                # Load MFCC (shape: 40, time)
                mfcc = np.load(npy_path)
                
                # Add Delta and Delta-Delta (shape becomes 120, time)
                mfcc_ext = add_deltas(mfcc)
                
                # Transpose to (time, 120)
                mfcc_ext = mfcc_ext.T
                
                # Zero-pad or truncate to fixed length (150)
                if mfcc_ext.shape[0] > MAX_LEN:
                    mfcc_ext = mfcc_ext[:MAX_LEN, :]
                else:
                    pad_width = MAX_LEN - mfcc_ext.shape[0]
                    mfcc_ext = np.pad(mfcc_ext, ((0, pad_width), (0, 0)), mode='constant')
                
                X.append(mfcc_ext)
                y.append(cls)
            except Exception as e:
                print(f"    [!] Error loading {os.path.basename(npy_path)}: {e}")

    X = np.array(X)
    y = np.array(y)
    
    # ── Standard Scaling ───────────────────────────────────────────
    # We need to flatten to scale correctly across all features
    num_samples, time_steps, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(num_samples, time_steps, num_features)
    
    # Save the scaler for use in the app
    os.makedirs(os.path.dirname(SCALER_OUT), exist_ok=True)
    joblib.dump(scaler, SCALER_OUT)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(CLASSES))
    
    return X, y_categorical, le

def build_hybrid_model(input_shape, num_classes):
    """Hybrid CNN-LSTM Model."""
    model = Sequential([
        Input(shape=input_shape),
        
        # CNN layers to extract spatial features from frequencies
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # LSTM layers to extract temporal patterns
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        
        # Fully connected layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Using a slightly lower learning rate for stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def run():
    # 1. Load and Preprocess
    X, y, le = load_data()
    
    if len(X) == 0:
        print("[!] No data loaded.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # 2. Build model
    model = build_hybrid_model(input_shape=X.shape[1:], num_classes=len(CLASSES))
    model.summary()
    
    # 3. Callbacks
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    
    # 4. Train
    print("\n🚀 Starting Advanced Training (Hybrid CNN-LSTM)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[lr_reducer, early_stop],
        verbose=1
    )
    
    # 5. Save Model
    model.save(MODEL_OUT)
    print(f"\n✅ V2 Model saved to: {MODEL_OUT}")
    print(f"✅ Scaler saved to: {SCALER_OUT}")
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📈 Final Test Accuracy: {accuracy*100:.2f}%")
    
    if accuracy < 0.70:
        print("💡 Hint: Data might be too noisy or augmentation is creating confusion. Try adding more real cat sounds.")

if __name__ == '__main__':
    run()
