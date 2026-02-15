
import os
import torch
import joblib
import pickle
from pathlib import Path
from torchvision import models, transforms
import torch.nn as nn

# Define Project Paths
# Script: backend/inference/model_loader.py
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # backend/inference -> backend -> project_root

# Model Paths - Using original models (fine-tuned models have incorrect class structure)
# TODO: Re-train fine-tuned models with correct data directory structure
IMAGE_MODEL_DIR = PROJECT_ROOT / "backend/trained_modelimages/image_model"
AUDIO_MODEL_DIR = PROJECT_ROOT / "backend/trained_modelimages/audio_model"

print(f"[Model Loader] Image Model Directory: {IMAGE_MODEL_DIR}")
print(f"[Model Loader] Audio Model Directory: {AUDIO_MODEL_DIR}")

class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.image_model = None
            cls._instance.audio_model = None
            cls._instance.audio_scaler = None
            cls._instance.audio_label_encoder = None
            cls._instance.image_classes = None
            cls._instance.device = torch.device("cpu") # User requested CPU optimization
            cls._instance.load_models()
        return cls._instance

    def load_models(self):
        print("Loading models...")
        self.load_image_model()
        self.load_audio_model()
        print("Models loaded.")

    def load_image_model(self):
        try:
            model_path = IMAGE_MODEL_DIR / "model.pth"
            classes_path = IMAGE_MODEL_DIR / "classes.txt"
            
            if not model_path.exists():
                print(f"Warning: Image model not found at {model_path}")
                return

            # Load Classes
            if classes_path.exists():
                with open(classes_path, 'r') as f:
                    self.image_classes = [line.strip() for line in f.readlines()]
            else:
                print("Warning: Image classes.txt not found. dynamic inference might fail.")
                self.image_classes = [] 

            # Initialize Model Architecture (ResNet50)
            self.image_model = models.resnet50(weights=None) # No need to download weights, we load state_dict
            num_ftrs = self.image_model.fc.in_features
            
            # Adjust final layer to match number of classes
            if self.image_classes:
                self.image_model.fc = nn.Linear(num_ftrs, len(self.image_classes))
            else:
                # Fallback if classes unknown (should not happen in prod)
                # Typically we need to know the class count to load state dict correctly if strict=True
                # Asking user to ensure training is done first.
                print("Error: Cannot load image model without knowing class count.")
                return

            # Load Weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.image_model.load_state_dict(state_dict)
            self.image_model.to(self.device)
            self.image_model.eval()
            print("Image model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading image model: {e}")

    def load_audio_model(self):
        try:
            model_path = AUDIO_MODEL_DIR / "audio_classifier.pkl"
            scaler_path = AUDIO_MODEL_DIR / "scaler.pkl"
            encoder_path = AUDIO_MODEL_DIR / "label_encoder.pkl"

            if not model_path.exists():
                 print(f"Warning: Audio model not found at {model_path}")
                 return
            
            self.audio_model = joblib.load(model_path)
            self.audio_scaler = joblib.load(scaler_path)
            self.audio_label_encoder = joblib.load(encoder_path)
            print("Audio model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading audio model: {e}")

# Global instance
loader = ModelLoader()
