You are a Senior ML Architect and MLOps Engineer.

GOAL:
Design and implement a complete local AI training pipeline using the local model:
- Model: gemma3:4b (running locally)
- Purpose: Distill knowledge into a smaller custom student model
- Data: Multimodal (image, audio, classified & non-classified data)
- Path: C:\Users\ramak\OneDrive\Desktop\project_smartx\project_mew\distling_model\user_data

REQUIREMENTS:

1. SYSTEM ARCHITECTURE
- Design an agentic AI pipeline with:
    - Data ingestion layer
    - Data analysis module
    - Train/test split module
    - Distillation training module
    - Human Feedback (RLHF-like) retraining loop
    - Evaluation engine (accuracy + confusion matrix + F1)
    - Model versioning system
    - Monitoring with Prometheus + Grafana (Docker-based, no login required)

2. PROJECT STRUCTURE
Create a clean 4–5 folder production-ready structure including:
- data_processing/
- training/
- distillation/
- feedback_loop/
- monitoring/
- frontend/
- backend/

Include explanation of what goes in each folder.

3. DISTILLATION DESIGN
- Use gemma3:4b as teacher model.
- Student model should:
    - Be smaller
    - Be optimized for classification
    - Learn from soft-label outputs (temperature scaling)
- Include:
    - Knowledge distillation loss formula
    - Hard label + soft label blending
    - Temperature parameter strategy
- Support multimodal inputs (audio + image)

4. HUMAN FEEDBACK LOOP
- Design a system to:
    - Save human corrections
    - Store them in feedback DB
    - Trigger retraining automatically
- Include:
    - Feedback weighting strategy
    - Incremental fine-tuning approach
    - Data validation pipeline

5. MONITORING
- Design Docker-based:
    - Prometheus metrics exporter
    - Grafana dashboard
- Track:
    - Accuracy
    - Loss
    - Drift detection
    - Feedback volume
    - Inference latency
- Grafana must start locally without authentication.

6. MODEL TRAINING FLOW
Explain:
- Initial supervised training
- Distillation phase
- Human feedback retraining phase
- Continuous improvement cycle

7. MULTIMEDIA SUPPORT
- Explain how to preprocess:
    - Images
    - Audio (MFCC or embeddings)
- How to align them with gemma teacher embeddings.

8. AGENTIC AI DESIGN
- Use modular AI agents:
    - Data Agent
    - Training Agent
    - Feedback Agent
    - Evaluation Agent
    - Monitoring Agent

Explain how agents communicate.

9. FRONTEND + BACKEND
- Backend: FastAPI
- Frontend: Cinematic studio-style UI
- Real-time prediction + feedback submission
- Model version display

10. OUTPUT FORMAT
Respond with:
- Clear architecture diagram (text form)
- Folder structure
- Training pipeline explanation
- Monitoring setup steps
- Distillation formula
- Feedback loop workflow
- Deployment steps

Be practical and production-level.
Do NOT simplify.we have virryal environent an it is avctiveted we can use to install librearies here onely (.venv) PS C:\Users\ramak\OneDrive\Desktop\project_smartx\project_mew\distling_model> and create requirement .txt file also 