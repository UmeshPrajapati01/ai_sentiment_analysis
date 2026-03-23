import os
import numpy as np
import librosa
import joblib
import cv2
from pathlib import Path

# ==========================================
# AUDIO CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(BASE_DIR).parent
TARGET_AUDIO_SR = 22050

# Path to persistent classic components
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "audio_scaler_classic.pkl")
audio_scaler = None

# ==========================================
# CAT FACE DETECTION (soft check only)
# ==========================================
# Load Haar cascade once at module level — never crashes if file missing
_cat_cascade = None
def _get_cascade():
    global _cat_cascade
    if _cat_cascade is None:
        try:
            xml_path = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
            _cat_cascade = cv2.CascadeClassifier(xml_path)
        except Exception:
            _cat_cascade = None
    return _cat_cascade

def detect_cat_face(image_path):
    """
    Soft check: returns True if a cat face is likely present, False otherwise.
    NEVER raises an exception — always returns a bool.
    Prediction always runs regardless of this result.
    """
    try:
        cascade = _get_cascade()
        if cascade is None or cascade.empty():
            return True  # cascade unavailable → assume OK, no warning

        img = cv2.imread(str(image_path))
        if img is None:
            return True  # can't read image → assume OK

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Try with relaxed params first (catches more faces)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            return True

        # Second pass — even more relaxed (catches partial/angled faces)
        faces2 = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=1,
            minSize=(20, 20)
        )
        return len(faces2) > 0

    except Exception:
        return True  # any error → assume OK, never block prediction

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
        S_dB = (S_dB + 80) / 80
        
        # Add channel and batch dimension for TensorFlow/Keras: (1, 64, 64, 1)
        spec_formatted = S_dB.reshape(1, 64, 64, 1).astype(np.float32)
        return spec_formatted
        
    except Exception as e:
        print(f"Error extracting Mel-Spec: {e}")
        return None

def extract_audio_features(file_path):
    """
    DEPRECATED. Kept for legacy RFC support.
    """
    try:
        y, sr = librosa.load(file_path, sr=TARGET_AUDIO_SR)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc
    except:
        return None
