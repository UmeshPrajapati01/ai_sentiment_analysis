import torch
import numpy as np
import librosa
from PIL import Image
from torchvision import transforms
from backend.inference.model_loader import loader
from backend.utils import extract_mel_spectrogram_sota # Use the 99% accuracy extractor

# Image Transforms (Must match training validation transforms)
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    if not loader.image_model:
        return "Model not loaded", 0.0
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = image_transforms(image).unsqueeze(0).to(loader.device)
        
        with torch.no_grad():
            outputs = loader.image_model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)
            
        class_idx = preds.item()
        conf_score = round(confidence.item() * 100, 2)  # e.g. 87.43

        if loader.image_classes and class_idx < len(loader.image_classes):
            return loader.image_classes[class_idx], conf_score
        return f"Class {class_idx}", conf_score
        
    except Exception as e:
        print(f"Image prediction error: {e}")
        return "Error", 0.0

def predict_audio(audio_path):
    """
    Predicts cat emotion from audio. 
    Switches dynamically between SOTA (CNN/Keras) and Legacy (RFC/Joblib).
    """
    if not loader.audio_model:
        return "Model not loaded", 0.0
        
    try:
        # ─── New SOTA High-Accuracy Path ─────────────────────────────────────
        if hasattr(loader.audio_model, 'predict') and not hasattr(loader.audio_model, 'predict_proba'):
            features = extract_mel_spectrogram_sota(audio_path)
            if features is None:
                print(f"[Predictor] Error: Spectrogram extraction failed for {audio_path}")
                return "Error (Processing)", 0.0
            
            preds = loader.audio_model.predict(features, verbose=0)
            class_idx = int(np.argmax(preds[0]))
            conf_score = round(float(np.max(preds[0])) * 100, 2)
            
            CLASSES = ['angry', 'fearful', 'happy', 'sad']
            if hasattr(loader, 'audio_label_encoder') and loader.audio_label_encoder:
                try:
                    return loader.audio_label_encoder.inverse_transform([class_idx])[0], conf_score
                except:
                    return CLASSES[class_idx] if class_idx < 4 else "Unknown", conf_score
            return (CLASSES[class_idx] if class_idx < 4 else "Unknown"), conf_score

        # ─── Legacy RFC Path (Backup) ──────────────────────────────────────────
        else:
            y, sr = librosa.load(audio_path, sr=22050, duration=3)
            if len(y) < 22050 * 3:
                y = np.pad(y, (0, int(22050 * 3) - len(y)))
                
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
            
            features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])
            features_scaled = loader.audio_scaler.transform([features])
            
            prediction_idx = loader.audio_model.predict(features_scaled)[0]
            # RFC confidence via predict_proba
            proba = loader.audio_model.predict_proba(features_scaled)[0]
            conf_score = round(float(np.max(proba)) * 100, 2)
            label = loader.audio_label_encoder.inverse_transform([prediction_idx])[0]
            return label, conf_score
        
    except Exception as e:
        import traceback
        print(f"[Predictor] CRITICAL ERROR: {e}")
        traceback.print_exc()
        return "Error", 0.0
