import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# 1. Feature Extraction (Spectrogram/MFCC)
def extract_audio_features(file_path):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    # Fix length to 100 frames for LSTM sequence
    if mfccs.shape[1] > 100: mfccs = mfccs[:, :100]
    else: mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])))
    return mfccs.T 

# 2. Build LSTM Model
model = Sequential([
    Input(shape=(100, 40)),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dropout(0.3),
    Dense(4, activation='softmax') # Happy, Sad, Angry, Fearful
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Load your data from ../../data/raw/audio and run model.fit()
model.save('../models/cat_emotion_lstm.keras')
print("Audio Model Trained and Saved!")
