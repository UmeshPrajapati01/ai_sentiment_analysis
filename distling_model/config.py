"""
Central Configuration for Distillation Pipeline
All paths, hyperparameters, and service endpoints in one place.
"""
import os

# ─── BASE PATHS ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
USER_DATA_DIR = os.path.join(PROJECT_ROOT, "user_data")
AUDIO_DATA_DIR = os.path.join(USER_DATA_DIR, "audio_data")
IMAGE_DATA_DIR = os.path.join(USER_DATA_DIR, "image_data")
CLASSIFIED_AUDIO_DIR = os.path.join(AUDIO_DATA_DIR, "classified_audio")
NON_CLASSIFIED_AUDIO_DIR = os.path.join(AUDIO_DATA_DIR, "non_classifiedaudio")
CLASSIFIED_IMAGE_DIR = os.path.join(IMAGE_DATA_DIR, "classified_imagedata")
NON_CLASSIFIED_IMAGE_DIR = os.path.join(IMAGE_DATA_DIR, "non_classifiedimagedata")
FEEDBACK_IMAGES_DIR = os.path.join(USER_DATA_DIR, "feedback_images")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# ─── DATABASE ─────────────────────────────────────────────────────────────────
DATABASE_URL = f"sqlite:///{os.path.join(USER_DATA_DIR, 'feedback.db')}"

# ─── AUDIO CLASSES (from classified_audio subdirectories) ─────────────────────
AUDIO_CLASSES = [
    "Angry", "Defense", "Fighting", "Happy", "HuntingMind",
    "Mating", "MotherCall", "Paining", "Resting", "Warning"
]
NUM_AUDIO_CLASSES = len(AUDIO_CLASSES)

# ─── IMAGE CLASSES (from classified_imagedata subdirectories) ─────────────────
IMAGE_CLASSES = ["Angry", "happy", "Sad", "Other"]
NUM_IMAGE_CLASSES = len(IMAGE_CLASSES)

# ─── COMBINED CLASSES (union of audio + image) ───────────────────────────────
ALL_CLASSES = sorted(list(set(AUDIO_CLASSES + IMAGE_CLASSES)))
NUM_CLASSES = len(ALL_CLASSES)

# ─── AUDIO PREPROCESSING ─────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SEC = 3.0          # pad/trim audio to this length
N_MFCC = 40                       # number of MFCC coefficients
N_MELS = 128                      # mel-spectrogram frequency bins
HOP_LENGTH = 512
AUDIO_MAX_LEN = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION_SEC)

# ─── IMAGE PREPROCESSING ─────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3

# ─── TRAINING HYPERPARAMETERS ────────────────────────────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50                        # aggressive: train longer for 80%+ accuracy
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42
DATA_AUGMENTATION = True                # enable on-the-fly augmentation

# ─── DISTILLATION HYPERPARAMETERS ────────────────────────────────────────────
TEMPERATURE = 4.0                  # softmax temperature for knowledge distillation
ALPHA = 0.5                        # blend ratio: alpha * hard_loss + (1-alpha) * soft_loss
DISTILLATION_LR = 5e-4
DISTILLATION_EPOCHS = 30            # aggressive: more distillation for deeper learning

# ─── TEACHER MODEL (Gemma3:4B via Ollama) ─────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_NAME = "gemma3:4b"
TEACHER_EMBEDDING_DIM = 768        # approximate dim for teacher embeddings

# ─── STUDENT MODEL ────────────────────────────────────────────────────────────
STUDENT_HIDDEN_DIM = 128
STUDENT_EMBEDDING_DIM = 128

# ─── FEEDBACK LOOP ────────────────────────────────────────────────────────────
FEEDBACK_RETRAIN_THRESHOLD = 20    # trigger retraining after N new feedback items
FEEDBACK_WEIGHT_MULTIPLIER = 1.5   # human-corrected samples are weighted higher
INCREMENTAL_LR = 1e-4              # lower LR for fine-tuning with feedback
INCREMENTAL_EPOCHS = 5

# ─── MONITORING ───────────────────────────────────────────────────────────────
PROMETHEUS_PORT = 8001             # port for metrics exporter
BACKEND_PORT = 8000
FRONTEND_PORT = 5500

# ─── MODEL VERSIONING ────────────────────────────────────────────────────────
MODEL_VERSION_FILE = os.path.join(MODELS_DIR, "version.json")

# Ensure critical directories exist
for d in [MODELS_DIR, LOGS_DIR, FEEDBACK_IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)
