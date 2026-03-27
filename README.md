<<<<<<< HEAD
# 🐱 MeowMood — AI Cat Emotion Recognition

A multimodal AI system that recognizes cat emotions from **facial images** and **audio (meow sounds)**.  
Built with PyTorch (vision), scikit-learn/TensorFlow (audio), and Flask (web app).

---

## Project Overview

| Modality | Model | Approach |
|----------|-------|----------|
| Image | ResNet50 (PyTorch) | Transfer learning, fine-tuned on cat face dataset |
| Audio | Random Forest + MFCC features | MFCC, Chroma, Mel-Spectrogram, Spectral Contrast |
| Fusion | Combined inference | Merges image + audio predictions at runtime |

Detected emotions: `Happy`, `Angry`, `Sad`, `Scared`, `Relaxed`, `Curious`, `Defense`, `Fighting`

---

## File Structure

```
ai_sentiment/
├── app.py                          # Flask app — all routes & API endpoints
├── requirements.txt                # Python dependencies
├── database/
│   └── database_logic.py           # SQLAlchemy models (User, Prediction), plans, credits
├── backend/
│   ├── utils.py                    # Audio feature extraction helpers
│   ├── inference/
│   │   ├── model_loader.py         # Loads image + audio models at startup
│   │   └── predictor.py            # predict_image() and predict_audio() functions
│   ├── models_training/
│   │   ├── image_model.py          # ResNet50 training pipeline
│   │   ├── audio_model.py          # Random Forest training pipeline
│   │   └── finetune_models.py      # Fine-tuning scripts
│   └── trained_modelimages/
│       ├── image_model/            # model.pth, classes.txt, training curves
│       ├── image_model_finetuned/  # Fine-tuned image model artifacts
│       ├── audio_model/            # audio_classifier.pkl, scaler.pkl, label_encoder.pkl
│       └── audio_model_finetuned/  # Fine-tuned audio model artifacts
├── front_end/
│   ├── templates/                  # Jinja2 HTML templates
│   └── static/                     # CSS, JS, images
└── colected_data/                  # Raw training datasets
    ├── audiofiles/                 # WAV files organized by emotion class
    └── imagefiles/                 # Cat face images organized by emotion class
```

---

## Setup & Installation

### 1. Clone / open the project
```bash
# Already in the project directory
cd ai_sentiment
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```
Open http://127.0.0.1:5000 in your browser.

---

## Demo Account

| Field | Value |
|-------|-------|
| Email | umesh.prajapati0506@gmail.com |
| Password | Umesh@2026 |
| Plan | Yearly Premium |
| Credits | 9999 |

---

## Training the Models

### Image Model (ResNet50)
```bash
python backend/models_training/image_model.py
```
- Input: `data_analysis/test_traindeddata/image_data/imagetraindata/`
- Output: `backend/trained_modelimages/image_model/`
  - `model.pth` — trained weights
  - `classes.txt` — class labels
  - `accuracy_curve.png`, `loss_curve.png` — training history

### Audio Model (Random Forest)
```bash
python backend/models_training/audio_model.py
```
- Input: `data_analysis/test_traindeddata/audio_data/train_data/`
- Output: `backend/trained_modelimages/audio_model/`
  - `audio_classifier.pkl` — trained model
  - `scaler.pkl` — feature scaler
  - `label_encoder.pkl` — label encoder
  - `classification_report.txt` — accuracy + F1-score per class
  - `confusion_matrix.png` — visual error analysis

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Redirect to dashboard or landing |
| GET/POST | `/dashboard` | Main analysis page (upload image/audio) |
| POST | `/fusion` | Multimodal fusion prediction (JSON) |
| GET | `/history` | Prediction history |
| GET | `/history/download` | Download history as CSV |
| GET | `/metrics` | Analytics & model performance |
| GET | `/api/chart-data` | Live chart data (JSON) |
| GET | `/plans` | Subscription plans |
| POST | `/subscribe/<plan>` | Subscribe to a plan |

---

## Model Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| Image (ResNet50) | ~92% | Validated on 20% holdout split |
| Audio (Random Forest) | ~87% | MFCC + Chroma + Mel features |

Metrics tracked: Accuracy, Precision, Recall, F1-score (per class).  
See `backend/trained_modelimages/audio_model/classification_report.txt` for full audio report.

---

## Tech Stack

- **Backend**: Python, Flask, SQLAlchemy, Flask-Login, Flask-Bcrypt
- **Vision AI**: PyTorch, torchvision (ResNet50)
- **Audio AI**: librosa, scikit-learn (Random Forest), TensorFlow/Keras
- **Frontend**: HTML/CSS/JS, Chart.js, Glassmorphism UI
- **Database**: SQLite

---

## References

- CNN-based cat facial expression recognition — [IJEAT](https://journals.blueeyesintelligence.org/index.php/ijeat/article/view/388)
- Cat sound emotion recognition — [JL-TFMSFNet, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417424014878)
- Data augmentation — [TensorFlow Docs](https://www.tensorflow.org/tutorials/images/data_augmentation)
- MFCC audio features — librosa documentation
=======
# Development-of-an-AI-Based-Cat-Emotion-Recognition-System-Using-Facial-and-Vocal-Analysis
The objective of this project is to develop an intelligent system capable of detecting cat emotions using both facial expressions and vocal sounds. The system is designed to help users understand pet behavior through AI-based analysis. The project aims to: • Build deep learning models for emotion detection • 
>>>>>>> e43a8a482d5c9c82adb68d7dcafcef8fa2438dc6
