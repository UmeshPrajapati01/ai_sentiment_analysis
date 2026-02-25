"""
FastAPI Backend - Cat Emotion Analysis
=======================================
Production API for the Distillation Pipeline.

Endpoints:
  POST /predict/image   -- Upload image for classification
  POST /predict/audio   -- Upload audio for classification
  POST /feedback        -- Submit human correction
  GET  /model/info      -- Current model version + metrics
  GET  /health          -- Health check
  GET  /                -- Frontend UI
"""

import os
import sys
import uuid
import time
import json
import logging
import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel
import shutil
import torch
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Emotion descriptions for each class
# ──────────────────────────────────────────────────────────────────────────

EMOTION_DESCRIPTIONS = {
    # Image classes
    "Angry": {
        "emoji": "😾",
        "message": "Your cat looks Angry!",
        "description": "Your cat appears to be irritated or upset. Look for flattened ears, dilated pupils, and a tense body posture. Give them some space and let them calm down.",
        "tips": "Avoid sudden movements and loud noises. Let your cat come to you when ready."
    },
    "happy": {
        "emoji": "😸",
        "message": "Your cat looks Happy!",
        "description": "Your cat is in a great mood! Look for relaxed ears, slow blinking, and possibly purring. This is the perfect time for gentle interaction.",
        "tips": "Enjoy some playtime or gentle petting. Your cat is feeling content and social."
    },
    "Sad": {
        "emoji": "😿",
        "message": "Your cat looks Sad.",
        "description": "Your cat appears to be feeling down or withdrawn. They might be hiding more, eating less, or showing less interest in play.",
        "tips": "Try offering their favorite treats or toys. If this persists, consider a vet visit."
    },
    "Other": {
        "emoji": "🐱",
        "message": "Your cat has a Neutral expression.",
        "description": "Your cat appears calm and relaxed with a neutral expression. They might be observing their surroundings or simply resting.",
        "tips": "Your cat is comfortable in their environment. A great time for bonding!"
    },
    # Audio classes
    "Defense": {
        "emoji": "🙀",
        "message": "Your cat sounds Defensive!",
        "description": "Your cat is making defensive vocalizations. They might feel threatened or cornered.",
        "tips": "Remove any perceived threats and give your cat a safe space to retreat to."
    },
    "Fighting": {
        "emoji": "😾",
        "message": "Your cat sounds like it's in a Fight!",
        "description": "Aggressive vocalizations detected. Your cat may be in a confrontation with another animal.",
        "tips": "Separate the animals carefully. Never put your hands between fighting cats."
    },
    "Happy": {
        "emoji": "😸",
        "message": "Your cat sounds Happy!",
        "description": "Your cat is making content, happy sounds. They might be purring, chirping, or making soft meows.",
        "tips": "Keep doing what you're doing! Your cat is enjoying the moment."
    },
    "HuntingMind": {
        "emoji": "🐾",
        "message": "Your cat is in Hunting Mode!",
        "description": "Your cat is making hunting vocalizations - chattering or chirping, usually directed at prey (birds, insects).",
        "tips": "Engage them with interactive toys to satisfy their hunting instinct."
    },
    "Mating": {
        "emoji": "💕",
        "message": "Your cat is making Mating Calls.",
        "description": "These vocalizations are associated with mating behavior. Common in unspayed/unneutered cats.",
        "tips": "Consider spaying/neutering if not already done. Consult your vet."
    },
    "MotherCall": {
        "emoji": "🍼",
        "message": "Your cat is making Mother Calls.",
        "description": "These are nurturing vocalizations, typically from a mother cat communicating with kittens.",
        "tips": "If your cat has kittens, ensure they have a warm, quiet nesting area."
    },
    "Paining": {
        "emoji": "🩹",
        "message": "Your cat sounds like it's in Pain!",
        "description": "Your cat is making distressed vocalizations that may indicate pain or discomfort.",
        "tips": "Check for visible injuries. If persistent, seek veterinary attention immediately."
    },
    "Resting": {
        "emoji": "😴",
        "message": "Your cat sounds Relaxed and Resting.",
        "description": "Soft, gentle vocalizations indicating your cat is comfortable and at rest.",
        "tips": "Let your sleeping cat lie! They're perfectly content."
    },
    "Warning": {
        "emoji": "⚠️",
        "message": "Your cat is giving a Warning!",
        "description": "Growling or hissing detected. Your cat is warning something to stay away.",
        "tips": "Identify what's causing the stress and remove it if possible."
    }
}


# ──────────────────────────────────────────────────────────────────────────
# APP INITIALIZATION
# ──────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cat Emotion Analyzer - AI Studio",
    description="Upload cat images or audio to analyze their emotions using AI",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────
# PROMETHEUS METRICS
# ──────────────────────────────────────────────────────────────────────────

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    )
    PREDICTIONS_TOTAL = Counter('predictions_total', 'Total predictions made',
                                ['modality'])
    PREDICTION_LATENCY = Histogram('prediction_latency_seconds',
                                    'Prediction latency in seconds',
                                    ['modality'])
    FEEDBACK_TOTAL = Counter('feedback_total', 'Total feedback submissions')
    MODEL_VERSION_GAUGE = Gauge('model_version', 'Current model version')
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────

class ModelManager:
    """Manages loading and inference for both audio and image models."""

    def __init__(self):
        self.audio_model = None
        self.image_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_version = 0
        self.image_version = 0
        self.audio_accuracy = 0.0
        self.image_accuracy = 0.0
        self.audio_loaded = False
        self.image_loaded = False

    def load_latest(self):
        """Load the latest trained audio and image models."""
        self._load_audio_model()
        self._load_image_model()

    def _load_audio_model(self):
        """Load latest audio model."""
        try:
            from training.models import AudioStudentModel

            # Try distilled first, fallback to supervised
            for model_type in ["audio_student_distilled", "audio_student_supervised"]:
                model, checkpoint = self._load_model_by_type(
                    AudioStudentModel, model_type,
                    num_classes=config.NUM_AUDIO_CLASSES,
                    n_mfcc=config.N_MFCC
                )
                if model is not None:
                    self.audio_model = model.to(self.device)
                    self.audio_model.eval()
                    self.audio_version = checkpoint.get('version', 0)
                    metrics = checkpoint.get('metrics', {})
                    self.audio_accuracy = metrics.get('accuracy', 0)
                    self.audio_loaded = True
                    logger.info(f"[OK] Audio model v{self.audio_version} loaded "
                                 f"(acc={self.audio_accuracy:.4f}, type={model_type})")
                    return
            logger.warning("[WARN] No audio model found -- audio in DEMO mode")
        except Exception as e:
            logger.warning(f"[WARN] Audio model load failed: {e}")

    def _load_image_model(self):
        """Load latest image model."""
        try:
            from training.models import ImageStudentModel

            for model_type in ["image_student_distilled", "image_student_supervised"]:
                model, checkpoint = self._load_model_by_type(
                    ImageStudentModel, model_type,
                    num_classes=config.NUM_IMAGE_CLASSES
                )
                if model is not None:
                    self.image_model = model.to(self.device)
                    self.image_model.eval()
                    self.image_version = checkpoint.get('version', 0)
                    metrics = checkpoint.get('metrics', {})
                    self.image_accuracy = metrics.get('accuracy', 0)
                    self.image_loaded = True
                    logger.info(f"[OK] Image model v{self.image_version} loaded "
                                 f"(acc={self.image_accuracy:.4f}, type={model_type})")
                    return
            logger.warning("[WARN] No image model found -- image in DEMO mode")
        except Exception as e:
            logger.warning(f"[WARN] Image model load failed: {e}")

    def _load_model_by_type(self, model_class, model_type, **model_kwargs):
        """Load the latest version of a specific model type from registry."""
        version_file = config.MODEL_VERSION_FILE
        if not os.path.exists(version_file):
            return None, None

        with open(version_file, 'r') as f:
            registry = json.load(f)

        matching = [m for m in registry["models"] if m["type"] == model_type]
        if not matching:
            return None, None

        # Get best model by accuracy (not just latest)
        best = max(matching, key=lambda x: x.get("accuracy", 0))
        filepath = os.path.join(config.MODELS_DIR, best["file"])

        if not os.path.exists(filepath):
            return None, None

        model = model_class(**model_kwargs)
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint

    def predict_audio(self, audio_path: str) -> dict:
        """Run inference on audio file. Returns full analysis report."""
        from data_processing import extract_mfcc
        import numpy as np

        start_time = time.time()

        mfcc = extract_mfcc(
            audio_path,
            sr=config.AUDIO_SAMPLE_RATE,
            n_mfcc=config.N_MFCC,
            max_len=config.AUDIO_MAX_LEN,
            hop_length=config.HOP_LENGTH
        )
        tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        if self.audio_loaded and self.audio_model is not None:
            with torch.no_grad():
                logits = self.audio_model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                confidence, predicted = probs.max(0)
                class_name = config.AUDIO_CLASSES[predicted.item()]
                conf = confidence.item()
                # All class probabilities
                all_probs = {config.AUDIO_CLASSES[i]: round(probs[i].item(), 4)
                             for i in range(len(config.AUDIO_CLASSES))}
            demo_mode = False
        else:
            import random
            class_name = random.choice(config.AUDIO_CLASSES)
            conf = round(random.uniform(0.5, 0.95), 4)
            all_probs = {c: round(random.uniform(0.01, 0.3), 4)
                         for c in config.AUDIO_CLASSES}
            all_probs[class_name] = conf
            demo_mode = True

        latency = time.time() - start_time

        if PROM_AVAILABLE:
            PREDICTIONS_TOTAL.labels(modality='audio').inc()
            PREDICTION_LATENCY.labels(modality='audio').observe(latency)

        # Build emotion details
        emotion = EMOTION_DESCRIPTIONS.get(class_name, {})

        return {
            "prediction": class_name,
            "confidence": round(conf, 4),
            "all_probabilities": all_probs,
            "emoji": emotion.get("emoji", "🐱"),
            "message": emotion.get("message", f"Detected: {class_name}"),
            "description": emotion.get("description", ""),
            "tips": emotion.get("tips", ""),
            "latency_ms": round(latency * 1000, 1),
            "model_version": self.audio_version,
            "model_accuracy": self.audio_accuracy,
            "demo_mode": demo_mode,
            "modality": "audio"
        }

    def predict_image(self, image_path: str) -> dict:
        """Run inference on image file. Returns full analysis report."""
        from data_processing import preprocess_image
        import numpy as np

        start_time = time.time()

        img = preprocess_image(image_path, size=config.IMAGE_SIZE)
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

        if self.image_loaded and self.image_model is not None:
            with torch.no_grad():
                logits = self.image_model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                confidence, predicted = probs.max(0)
                class_name = config.IMAGE_CLASSES[predicted.item()]
                conf = confidence.item()
                all_probs = {config.IMAGE_CLASSES[i]: round(probs[i].item(), 4)
                             for i in range(len(config.IMAGE_CLASSES))}
            demo_mode = False
        else:
            import random
            class_name = random.choice(config.IMAGE_CLASSES)
            conf = round(random.uniform(0.5, 0.95), 4)
            all_probs = {c: round(random.uniform(0.01, 0.3), 4)
                         for c in config.IMAGE_CLASSES}
            all_probs[class_name] = conf
            demo_mode = True

        latency = time.time() - start_time

        if PROM_AVAILABLE:
            PREDICTIONS_TOTAL.labels(modality='image').inc()
            PREDICTION_LATENCY.labels(modality='image').observe(latency)

        emotion = EMOTION_DESCRIPTIONS.get(class_name, {})

        return {
            "prediction": class_name,
            "confidence": round(conf, 4),
            "all_probabilities": all_probs,
            "emoji": emotion.get("emoji", "🐱"),
            "message": emotion.get("message", f"Detected: {class_name}"),
            "description": emotion.get("description", ""),
            "tips": emotion.get("tips", ""),
            "latency_ms": round(latency * 1000, 1),
            "model_version": self.image_version,
            "model_accuracy": self.image_accuracy,
            "demo_mode": demo_mode,
            "modality": "image"
        }


# Global model manager
model_mgr = ModelManager()


# ──────────────────────────────────────────────────────────────────────────
# FEEDBACK DATABASE
# ──────────────────────────────────────────────────────────────────────────

from feedback_loop import FeedbackDB, DataValidator

feedback_db = FeedbackDB()
data_validator = DataValidator(
    valid_classes=config.AUDIO_CLASSES + config.IMAGE_CLASSES,
    data_dir=config.USER_DATA_DIR
)


# ──────────────────────────────────────────────────────────────────────────
# STARTUP EVENT
# ──────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Load models on startup."""
    os.makedirs(config.FEEDBACK_IMAGES_DIR, exist_ok=True)
    model_mgr.load_latest()


# ──────────────────────────────────────────────────────────────────────────
# API ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "audio_model_loaded": model_mgr.audio_loaded,
        "image_model_loaded": model_mgr.image_loaded,
        "audio_model_version": model_mgr.audio_version,
        "image_model_version": model_mgr.image_version,
        "device": str(model_mgr.device),
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    """Upload and classify an audio file."""
    file_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1] if "." in file.filename else "wav"
    saved_filename = f"{file_id}.{ext}"
    file_path = os.path.join(config.FEEDBACK_IMAGES_DIR, saved_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = model_mgr.predict_audio(file_path)

    feedback_db.save_feedback(
        record_id=file_id,
        filename=saved_filename,
        modality="audio",
        prediction=result["prediction"],
        confidence=result["confidence"]
    )

    result["id"] = file_id
    return result


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Upload and classify an image file."""
    file_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    saved_filename = f"{file_id}.{ext}"
    file_path = os.path.join(config.FEEDBACK_IMAGES_DIR, saved_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = model_mgr.predict_image(file_path)

    feedback_db.save_feedback(
        record_id=file_id,
        filename=saved_filename,
        modality="image",
        prediction=result["prediction"],
        confidence=result["confidence"]
    )

    result["id"] = file_id
    return result


@app.post("/feedback")
async def submit_feedback(file_id: str = Form(...), correct_label: str = Form(...)):
    """Submit human correction for a prediction."""
    is_valid, error = data_validator.validate_feedback(correct_label)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    success = feedback_db.update_human_label(file_id, correct_label)
    if not success:
        raise HTTPException(status_code=404, detail="Prediction record not found")

    if PROM_AVAILABLE:
        FEEDBACK_TOTAL.inc()

    return {
        "status": "success",
        "message": f"Feedback saved for {file_id}",
        "should_retrain": feedback_db.should_retrain(),
        "unused_feedback": feedback_db.get_unused_count(),
        "threshold": config.FEEDBACK_RETRAIN_THRESHOLD
    }


@app.get("/model/info")
async def model_info():
    """Get current model version and metrics."""
    version_data = {}
    if os.path.exists(config.MODEL_VERSION_FILE):
        with open(config.MODEL_VERSION_FILE, 'r') as f:
            version_data = json.load(f)

    return {
        "audio": {
            "loaded": model_mgr.audio_loaded,
            "version": model_mgr.audio_version,
            "accuracy": model_mgr.audio_accuracy,
            "classes": config.AUDIO_CLASSES,
            "num_classes": config.NUM_AUDIO_CLASSES,
        },
        "image": {
            "loaded": model_mgr.image_loaded,
            "version": model_mgr.image_version,
            "accuracy": model_mgr.image_accuracy,
            "classes": config.IMAGE_CLASSES,
            "num_classes": config.NUM_IMAGE_CLASSES,
        },
        "device": str(model_mgr.device),
        "teacher_model": config.OLLAMA_MODEL_NAME,
        "version_history": version_data.get("models", [])
    }


# Prometheus metrics endpoint
if PROM_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


# ──────────────────────────────────────────────────────────────────────────
# SERVE FRONTEND
# ──────────────────────────────────────────────────────────────────────────

frontend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(
            content=content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )
    return HTMLResponse(content="<h1>Frontend not found. Place index.html in /frontend/</h1>")

# MIME type map for static files
MIME_TYPES = {
    ".css": "text/css",
    ".js": "application/javascript",
    ".html": "text/html",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
}

@app.get("/static/{filename:path}")
async def serve_static(filename: str):
    """Serve CSS, JS, and other static assets from frontend/static/"""
    # Strip query params (cache busting e.g. ?v=2)
    clean_name = filename.split("?")[0] if "?" in filename else filename
    filepath = os.path.join(frontend_dir, "static", clean_name)
    if os.path.exists(filepath):
        ext = os.path.splitext(clean_name)[1].lower()
        media_type = MIME_TYPES.get(ext)
        return FileResponse(filepath, media_type=media_type)
    # Fallback: check frontend root
    filepath_root = os.path.join(frontend_dir, clean_name)
    if os.path.exists(filepath_root):
        ext = os.path.splitext(clean_name)[1].lower()
        media_type = MIME_TYPES.get(ext)
        return FileResponse(filepath_root, media_type=media_type)
    raise HTTPException(status_code=404, detail=f"Static file not found: {filename}")

@app.get("/images_videos/{filename:path}")
async def serve_media(filename: str):
    """Serve images and videos from frontend/images_videos/"""
    filepath = os.path.join(frontend_dir, "images_videos", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="File not found")


# ──────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.BACKEND_PORT, reload=False)
