import os
import joblib
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

# Fix relative imports
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(BASE_DIR).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Paths
IMAGE_MODEL_DIR = PROJECT_ROOT / "backend" / "trained_modelimages" / "image_model"
AUDIO_MODEL_DIR = PROJECT_ROOT / "backend" / "trained_modelimages" / "audio_model"

class ModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_model = None
        self.image_classes = None
        self.audio_model = None
        self.audio_scaler = None
        self.audio_label_encoder = None
        self.load_models()

    def load_models(self):
        self.load_image_model()
        self.load_audio_model()

    def load_image_model(self):
        try:
            model_path = IMAGE_MODEL_DIR / "model.pth"
            classes_path = IMAGE_MODEL_DIR / "classes.txt"
            
            if not model_path.exists():
                print(f"Warning: Image model not found at {model_path}")
                return
            
            # Load classes
            with open(classes_path, 'r') as f:
                self.image_classes = [line.strip() for line in f.readlines()]
            
            # Recreate architecture
            self.image_model = models.resnet50(weights=None)
            num_ftrs = self.image_model.fc.in_features
            self.image_model.fc = nn.Linear(num_ftrs, len(self.image_classes))
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.image_model.load_state_dict(state_dict)
            self.image_model.to(self.device).eval()
            print("Image model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading image model: {e}")

    def load_audio_model(self):
        try:
            # 1. Try SOTA v2 High-Accuracy Model (Keras) - 99%+ Target
            sota_path = PROJECT_ROOT / "models" / "cat_emotion_sota_v2.keras"
            # Fallback path if stored in audio_model dir
            alt_sota_path = AUDIO_MODEL_DIR / "audio_classifier_sota.keras"
            
            if sota_path.exists():
                print(f"Loading SOTA Audio Model: {sota_path}")
                import tensorflow as tf
                self.audio_model = tf.keras.models.load_model(sota_path)
            elif alt_sota_path.exists():
                print(f"Loading SOTA Audio Model: {alt_sota_path}")
                import tensorflow as tf
                self.audio_model = tf.keras.models.load_model(alt_sota_path)
            else:
                # 2. Try Legacy RFC (Joblib)
                model_path = AUDIO_MODEL_DIR / "audio_classifier.pkl"
                scaler_path = AUDIO_MODEL_DIR / "scaler.pkl"
                encoder_path = AUDIO_MODEL_DIR / "label_encoder.pkl"

                if not model_path.exists():
                     print(f"Warning: No Audio model found at {model_path} or {sota_path}")
                     return
                
                self.audio_model = joblib.load(model_path)
                self.audio_scaler = joblib.load(scaler_path)
                self.audio_label_encoder = joblib.load(encoder_path)
                print("Legacy RFC Audio model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading audio model: {e}")

loader = ModelLoader()
