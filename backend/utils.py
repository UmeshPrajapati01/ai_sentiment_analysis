"""
utils.py
Helper module for backend inference. Contains functions for preprocessing
uploaded images and audio files before passing them to the models.
"""

import os
import cv2
import librosa
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import noisereduce as nr

# Constants
TARGET_IMAGE_SIZE = (224, 224)
TARGET_AUDIO_SR = 16000
MAX_PAD_LEN = 150

# Load Haar Cascade for cat face detection
# Ensure this OpenCV data path is correct for the deployment environment
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
cat_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def preprocess_image(image_path):
    """
    Reads an image, detects a cat face, crops, resizes to 224x224, 
    normalizes, and prepares it for CNN inference.
    
    Raises:
        ValueError: If image is unreadable or no cat face is detected.
    """
    # 1. Load image in BGR format
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image file. Could not read the image.")
        
    # 2. Detect face (use grayscale for detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply Haar Cascade for Cat Face Detection
    if not cat_cascade.empty():
        faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
    else:
        raise RuntimeError("Haar cascade file for cat face not loaded.")
        
    # Check if a cat face was detected
    if len(faces) == 0:
        raise ValueError("No cat face detected in the image. Please upload a clear photo of a cat's face.")
        
    # 4. Crop to the first detected cat face
    (x, y, w, h) = faces[0]
    cropped_face = img[y:y+h, x:x+w]
    
    # 5. Convert BGR to RGB
    img_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    
    # 6. Resize to matching model target size (224x224)
    resized = cv2.resize(img_rgb, TARGET_IMAGE_SIZE)
    
    # 7. Normalize pixel values (0-1)
    img_array = img_to_array(resized) / 255.0
    
    # 8. Add batch dimension for inference: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

import joblib # for scaler loading

# Scaler path (absolute)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALER_PATH = os.path.join(BASE_DIR, "models", "audio_scaler_classic.pkl")
audio_scaler = None

def get_scaler():
    global audio_scaler
    if audio_scaler is None:
        if os.path.exists(SCALER_PATH):
            audio_scaler = joblib.load(SCALER_PATH)
    return audio_scaler

def extract_mel_spectrogram_sota(file_path):
    """
    Extracts a 64x64 Mel-Spectrogram image exactly matching 
    the SOTA v2 training logic for 99% accuracy.
    """
    try:
        # Load at 16000Hz (Training standard)
        y, sr = librosa.load(file_path, sr=16000)
        
        # 1. Mel Spectrogram (64 bands)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=256)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 2. Resizing to (64, 64) - Padding or Truncating
        if S_dB.shape[1] > 64:
            S_dB = S_dB[:, :64]
        else:
            S_dB = np.pad(S_dB, ((0, 0), (0, 64 - S_dB.shape[1])), mode='constant')
            
        # 3. Global Normalization (Map -80dB..0dB to 0..1 scale)
        # matches train_audio_sota_v2.py line 40
        S_dB = (S_dB + 80) / 80
        
        # Add channel and batch dimension for TensorFlow/Keras: (1, 64, 64, 1)
        spec_formatted = S_dB.reshape(1, 64, 64, 1)
        return spec_formatted
        
    except Exception as e:
        print(f"Error extracting Mel-Spec: {e}")
        return None

def extract_audio_features(file_path):
    """
    DEPRECATED (Classic RFC features). 
    Kept for legacy support, but new SOTA models should use extract_mel_spectrogram_sota.
    """
    try:
        y, sr = librosa.load(file_path, sr=TARGET_AUDIO_SR)
        
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
        
        # Combine
        features = np.concatenate([
            mfcc_mean, mfcc_std, 
            chroma_mean, 
            mel_mean, 
            [np.mean(cent), np.mean(flat), np.mean(zcr)]
        ])
        
        # Scale
        scaler = get_scaler()
        if scaler:
            features = scaler.transform([features])
            
        return features # Shape: (1, 183)
        
    except Exception as e:
        raise ValueError(f"Error extracting features: {str(e)}")
