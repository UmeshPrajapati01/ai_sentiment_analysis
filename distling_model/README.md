# 🧬 DISTILL AI STUDIO
## Multimodal Knowledge Distillation Pipeline

A production-grade, agentic AI pipeline that distills knowledge from **Gemma3:4B** (teacher) into a smaller, optimized student model for multimodal classification (audio + image).

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     DISTILL AI STUDIO - ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌────────────────┐    ┌─────────────────────────┐  │
│  │  DATA AGENT   │───▶│ TRAINING AGENT │───▶│   EVALUATION AGENT     │  │
│  │              │    │                │    │                         │  │
│  │ • Ingest     │    │ • Supervised   │    │ • Accuracy / F1        │  │
│  │ • Validate   │    │ • Distillation │    │ • Confusion Matrix     │  │
│  │ • Split      │    │ • Versioning   │    │ • Classification Rpt   │  │
│  └──────┬───────┘    └───────┬────────┘    └────────────┬────────────┘  │
│         │                    │                           │               │
│         │            ┌───────▼────────┐                  │               │
│         │            │  MESSAGE BUS   │◀─────────────────┘               │
│         │            │  (Event Queue) │                                  │
│         │            └───────┬────────┘                                  │
│         │                    │                                           │
│  ┌──────▼───────┐    ┌───────▼────────┐    ┌─────────────────────────┐  │
│  │ FEEDBACK     │    │  MONITORING    │    │   GEMMA 3:4B (TEACHER)  │  │
│  │   AGENT      │    │    AGENT       │    │                         │  │
│  │ • Collect    │    │ • Prometheus   │    │ • Ollama (local)        │  │
│  │ • Validate   │    │ • Grafana      │    │ • Soft-label generation │  │
│  │ • Retrain    │    │ • Drift detect │    │ • Classification assist │  │
│  └──────────────┘    └────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
distling_model/
├── config.py                    # Central configuration (all paths, hyperparams)
├── main.py                      # CLI entry point (train/serve/agent/status)
├── agents.py                    # Agentic AI system (5 agents + message bus)
├── requirements.txt             # Python dependencies
│
├── data_processing/             # DATA INGESTION + PREPROCESSING
│   └── __init__.py              # MFCC extraction, image preprocessing,
│                                # dataset classes, train/val/test split
│
├── training/                    # MODEL TRAINING
│   ├── __init__.py
│   ├── models.py                # Student model architectures (Audio/Image/Multimodal)
│   └── pipeline.py              # Training pipeline (supervised + distillation + eval)
│
├── distillation/                # KNOWLEDGE DISTILLATION
│   └── __init__.py              # DistillationLoss, GemmaTeacher, MockTeacher,
│                                # TemperatureScheduler
│
├── feedback_loop/               # HUMAN FEEDBACK (RLHF-like)
│   └── __init__.py              # FeedbackDB, DataValidator, FeedbackWeighter,
│                                # Incremental fine-tuning
│
├── monitoring/                  # PROMETHEUS + GRAFANA
│   ├── docker-compose.yml       # Docker services
│   ├── prometheus.yml           # Scrape configuration
│   ├── metrics.py               # Prometheus metrics definitions
│   └── grafana/
│       ├── provisioning/        # Auto-provisioned datasources + dashboards
│       └── dashboards/          # Pre-built Grafana dashboard JSON
│
├── frontend/                    # CINEMATIC UI
│   └── index.html               # Studio-style web interface
│
├── backend/                     # FASTAPI SERVER
│   └── main.py                  # REST API (predict/feedback/model info/metrics)
│
├── models/                      # SAVED MODELS + VERSION REGISTRY
│   └── version.json             # Model version tracking
│
├── user_data/                   # YOUR DATA
│   ├── audio_data/
│   │   ├── classified_audio/    # 10 classes × 10 samples
│   │   └── non_classifiedaudio/ # 483 unlabeled .wav files
│   ├── image_data/
│   │   ├── classified_imagedata/
│   │   └── non_classifiedimagedata/
│   └── feedback_images/         # Uploaded prediction files
│
└── logs/                        # Training logs
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Activate your virtual environment
.\.venv\Scripts\activate

# Install all packages
pip install -r requirements.txt
```

### 2. Check System Status
```bash
python main.py status
```

### 3. Run Training Pipeline
```bash
python main.py train
```
This executes:
- Phase 1: Supervised training on classified audio data
- Phase 2: Knowledge distillation (uses Gemma3:4B if available, otherwise MockTeacher)
- Evaluation with accuracy, F1, confusion matrix
- Model versioning and saving

### 4. Start the API Server
```bash
python main.py serve
```
- **API**: http://localhost:8000
- **Frontend**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics

### 5. Start Monitoring (Docker Required)
```bash
cd monitoring
docker-compose up -d
```
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (no login required!)

### 6. Run Agentic Pipeline
```bash
python main.py agent
```

---

## 🔬 Knowledge Distillation

### Loss Formula
```
L_total = α · L_hard + (1 - α) · T² · L_soft

Where:
  L_hard = CrossEntropy(student_logits, true_labels)
  L_soft = KLDivergence(log_softmax(student/T), softmax(teacher/T))
  T      = Temperature (controls distribution softness)
  α      = Blending coefficient (default: 0.5)
```

### Temperature Strategy
| Phase | Temperature | Reasoning |
|-------|------------|-----------|
| Early Training | T=5.0 | Maximum inter-class knowledge transfer |
| Mid Training | T=3.0 | Balanced soft/hard signal |
| Late Training | T=1.5 | Focus on decision boundaries |
| **Schedule** | **Cosine Annealing** | **5.0 → 1.5 over training** |

### Teacher Model
- **Gemma3:4B** running locally via **Ollama**
- Generates soft-label probability distributions
- If unavailable, **MockTeacher** simulates teacher behavior for testing

---

## 🧠 Student Model Architecture

### AudioStudentModel (Primary)
```
Input: MFCC (1, 40, time_steps)
  → Conv2d(1→32) + BN + ReLU + MaxPool
  → Conv2d(32→64) + BN + ReLU + MaxPool
  → Conv2d(64→128) + BN + ReLU + AdaptivePool
  → Linear(2048→256) + ReLU + Dropout
  → Linear(256→128) + ReLU
  → Linear(128→10)  [10 emotion classes]
```

### Classes
`Angry | Defense | Fighting | Happy | HuntingMind | Mating | MotherCall | Paining | Resting | Warning`

---

## 🔄 Human Feedback Loop

### Workflow
```
User uploads file → Model predicts → User corrects (optional)
                                          ↓
                                    Feedback DB (SQLite)
                                          ↓
                              Count >= 20? → Trigger Retraining
                                          ↓
                                 Incremental Fine-tuning
                                 (weighted, lower LR)
                                          ↓
                                 New Model Version Saved
```

### Weighting Strategy
| Correction Type | Weight | Reasoning |
|----------------|--------|-----------|
| Model was WRONG | 1.5× | Prioritize learning from mistakes |
| Model was RIGHT | 1.0× | Standard reinforcement |
| Recent feedback | Higher | Recency decay factor (0.95^days) |

---

## 📊 Monitoring

### Prometheus Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `training_loss` | Gauge | Current training loss |
| `training_accuracy` | Gauge | Current training accuracy |
| `validation_accuracy` | Gauge | Validation set accuracy |
| `epochs_completed_total` | Counter | Total epochs completed |
| `predictions_total` | Counter | Total predictions (by modality) |
| `prediction_latency_seconds` | Histogram | Inference latency |
| `feedback_total` | Counter | Total feedback submissions |
| `model_version` | Gauge | Current model version |
| `model_accuracy` | Gauge | Latest model accuracy |
| `drift_score` | Gauge | Distribution drift score |

### Grafana Dashboard
Pre-configured with:
- Training accuracy gauge
- Loss curves (train + validation)
- Prediction latency histogram
- Feedback submission rate
- Model version tracker

---

## 🤖 Agentic AI System

### Agent Communication
Agents communicate via an **event-based Message Bus**:

| Agent | Responsibilities | Publishes | Listens To |
|-------|-----------------|-----------|------------|
| **DataAgent** | Ingest, validate, split | DATA_READY, DATA_ANALYZED | — |
| **TrainingAgent** | Supervised + distillation | TRAINING_COMPLETE, EVALUATION_REQUEST | DATA_READY |
| **EvaluationAgent** | Accuracy, F1, CM | EVALUATION_COMPLETE | EVALUATION_REQUEST |
| **FeedbackAgent** | Collect, validate, retrain | FEEDBACK_RECEIVED, RETRAIN_TRIGGER | — |
| **MonitoringAgent** | Prometheus, drift | METRICS_UPDATE | All events |

---

## 🖥️ Frontend + Backend

### Backend (FastAPI)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/audio` | POST | Upload audio for classification |
| `/predict/image` | POST | Upload image for classification |
| `/feedback` | POST | Submit human correction |
| `/model/info` | GET | Model version + metrics |
| `/feedback/stats` | GET | Feedback statistics |
| `/dashboard` | GET | Full dashboard data |
| `/metrics` | GET | Prometheus metrics |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger API docs |

### Frontend
- Cinematic dark theme with glassmorphism
- Drag-and-drop file upload (audio + image)
- Real-time prediction with confidence bars
- Human feedback submission dropdown
- Model version history display
- Architecture diagram
- Monitoring dashboard with external links

---

## 🎯 Training Pipeline Flow

```
1. SUPERVISED TRAINING (Phase 1)
   ├── Load classified audio data (10 classes × 10 samples)
   ├── Extract MFCC features (40 coefficients, 3s duration)
   ├── Train/Val/Test split (80/10/10)
   ├── Train AudioStudentModel with CrossEntropyLoss
   ├── ReduceLROnPlateau scheduler
   └── Save best model by validation accuracy

2. DISTILLATION (Phase 2)
   ├── Generate teacher soft-labels (Gemma3:4B or MockTeacher)
   ├── Temperature-scheduled distillation
   ├── Combined loss: α·CE + (1-α)·T²·KL
   └── Cosine temperature annealing: 5.0 → 1.5

3. HUMAN FEEDBACK (Phase 3 — ongoing)
   ├── Collect corrections via API
   ├── Validate label quality
   ├── Weighted incremental fine-tuning
   └── New model version on each retrain

4. EVALUATION
   ├── Accuracy, F1 (macro + weighted)
   ├── Confusion matrix
   ├── Per-class classification report
   └── Prometheus metrics export
```

---

## 🐳 Deployment Steps

### Step 1: Train the Model
```bash
python main.py train
```

### Step 2: Start the API
```bash
python main.py serve
```

### Step 3: Start Monitoring
```bash
cd monitoring
docker-compose up -d
```

### Step 4: Open the UI
- Frontend: http://localhost:8000/
- Grafana: http://localhost:3000/
- API Docs: http://localhost:8000/docs

### Step 5: Upload Files & Submit Feedback
Use the web interface to classify audio/images and correct predictions.

---

## ⚙️ Configuration

All settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `BATCH_SIZE` | 8 | Training batch size |
| `LEARNING_RATE` | 1e-3 | Initial learning rate |
| `NUM_EPOCHS` | 20 | Supervised training epochs |
| `TEMPERATURE` | 4.0 | Distillation temperature |
| `ALPHA` | 0.5 | Hard/soft loss blend ratio |
| `DISTILLATION_EPOCHS` | 15 | Distillation epochs |
| `FEEDBACK_RETRAIN_THRESHOLD` | 20 | Feedback items before retrain |
| `AUDIO_SAMPLE_RATE` | 16000 | Audio resampling rate |
| `N_MFCC` | 40 | MFCC coefficients |

---

*Built with PyTorch, FastAPI, Prometheus, Grafana, and Gemma3:4B* 🧬
