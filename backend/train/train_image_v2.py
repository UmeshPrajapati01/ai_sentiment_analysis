import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import sys

# Add parent dir to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import preprocess_image

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/raw/images"))
MODEL_SAVE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../models/cat_emotion_cnn.keras"))
CLASSES = ['angry', 'fearful', 'happy', 'sad']


def load_data():
    X = []
    y = []
    
    for label in CLASSES:
        folder = os.path.join(DATA_PATH, label)
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} not found.")
            continue
            
        print(f"Processing {label}...")
        for file in os.listdir(folder):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(folder, file)
                try:
                    # Use the same extraction logic as inference
                    feat = preprocess_image(path) # Shape: (1, 224, 224, 3)
                    X.append(feat[0]) 
                    y.append(label)
                except Exception as e:
                    # If face detection fails, try a simple resize fallback for training if needed
                    # but for this project we want to train ON the faces.
                    # Printing skip for now.
                    print(f"Skipping {path}: {e}")
                    
    return np.array(X), np.array(y)

print("--- Starting Image Model Training ---")
X, y = load_data()

if len(X) == 0:
    print("Error: No data found or face detection failed for all images.")
    sys.exit(1)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=len(CLASSES))

# Build Model (Target size 224x224 as per utils.py)
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"Training on {len(X)} samples...")
# Only 40 samples roughly, so few epochs or small batch
model.fit(X, y_categorical, epochs=30, batch_size=4, verbose=1)

# Ensure models directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

model.save(MODEL_SAVE_PATH)
print(f"Image model saved to {MODEL_SAVE_PATH}")
