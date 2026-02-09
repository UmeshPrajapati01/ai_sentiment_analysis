# MeowMood — 4-Week Weekly/Daily Teaching Plan (Mon–Fri, 2 hours/day)

## Overview
This 4-week, instructor-led course (2 hours/day) guides students through the MeowMood project: image and audio pipelines to recognize cat emotions. Each week ends with a deliverable. Students are expected to be comfortable with Python and basic deep learning tooling.

**Assumptions**
- Student level: Advanced (familiar with DL tooling).
- Session length: 2 hours per day (Mon–Fri).
- Weekly deliverable required.

**Key repository files (instructor reference)**
- `backend/models_training/image_model/cnnarchitecture.py` — vision training & evaluation (A5–A8)
- `backend/models_training/image_model/imagenornalization.py` — image preprocessing & augmentation (A1–A4)
- `backend/models_training/audio_model/audiocleaning.py` — audio cleaning & feature extraction
- `backend/models_training/audio_model/rnnlstm_archeitecture.py` — audio modeling (RNN/LSTM)
- `requirements.txt` and `README.md`

---

## Week 1 — Data & Preprocessing (Deliverable: cleaned image subset + preprocessing scripts)

- Monday: Kickoff & environment setup (2h)
  - Project goals, evaluation metrics (accuracy/F1), repo layout.
  - Create virtualenv, install `requirements.txt`, verify GPU availability.

- Tuesday: Image dataset acquisition (A1) (2h)
  - Scraping/curation strategies, label schema. Inspect `uncleaned_datasets/cleaned_data/cat_classifieddataimage`.
  - Exercise: create small sample downloader or sampling script.

- Wednesday: Image preprocessing (A2) (2h)
  - Resize, grayscale decision, normalization (ImageNet stats).
  - Run and inspect `imagenornalization.py` on a small subset.

- Thursday: Augmentation (A3) (2h)
  - Rotation, flip, brightness, color jitter; save augmented examples.

- Friday: EDA & class-balance strategy (A4) (2h)
  - Visualize class counts, sample images, choose balancing strategy (WeightedRandomSampler or class weights).
  - Deliverable: cleaned subset + EDA plots.

---

## Week 1 — Vision Model Design & Training (Deliverable: trained vision model + report)

- Monday: CNN architecture (A5) (2h)
  - Transfer learning concepts; compare MobileNetV2 vs ResNet18.
  - Walk through `cnnarchitecture.py` model construction.

- Tuesday: Training fundamentals (A6) (2h)
  - Dataloaders, stratified split, loss, optimizer, scheduler, early stopping.
  - Run a short smoke training (epochs=2, batch=8) to verify pipeline.

- Wednesday: Hyperparameter tuning (A7) (2h)
  - LR, dropout, grid vs random search. Run small grid-search in `cnnarchitecture.py` with reduced settings.

- Thursday: Evaluation (A8) (2h)
  - Confusion matrix, precision/recall/F1, per-class analysis. Use `evaluate_model` in `cnnarchitecture.py`.

- Friday: Packaging & demo (2h)
  - Save best weights, write an inference script to load `cat_emotion_best.pth` and run predictions.
  - Deliverable: trained model, evaluation report, short inference demo.

---

## Week 2 — Audio: Acquisition → Features → Cleaning (Deliverable: features CSV + EDA)

- Monday: Audio collection & label mapping (2h)
  - Recording best practices; create `label_map.json` to map fine-grained audio labels to the 4 target emotions.

- Tuesday: Audio cleaning (2h)
  - Silence trimming, noise reduction, resampling. Run `audiocleaning.py` on samples.

- Wednesday: Feature extraction (2h)
  - MFCCs, spectrograms, chroma, RMS. Save features to `output_weekwise/week1output/audio/features.csv`.

- Thursday: Audio EDA (2h)
  - Plot spectrograms, pitch distributions, class-wise feature histograms.

- Friday: Feature prep & selection planning (2h)
  - Scale features, split train/val, plan RF-based feature selection.

---

## Week 2 — Audio Modeling, Selection, Integration & Final Presentations (Deliverable: audio model + integrated demo & report)

- Monday: RNN/LSTM design (2h)
  - Sequence shapes, padding, LSTM layers; review `rnnlstm_archeitecture.py`.

- Tuesday: Train audio model (2h)
  - Train with reduced epochs; evaluate and save `best_model.h5` (or `.pth`).

- Wednesday: Feature selection (2h)
  - Use RandomForest importance or recursive selection; evaluate impact on performance.

- Thursday: Integration & demo (2h)
  - Late fusion: average/weighted ensemble of vision + audio models; build a simple inference demo script.

- Friday: Presentations & wrap-up (2h)
  - Students present deliverables, submit final report, discuss next steps.

---

## Quick verification & demo commands (use reduced settings for classroom)

Vision smoke test (short):
```bash
python backend/models_training/image_model/cnnarchitecture.py \
  --data uncleaned_datasets/cleaned_data/cat_classifieddataimage \
  --out output_weekwise/week2output/image --epochs 2 --batch 8 --base mobilenetv2
```

Audio feature extraction (example):
```bash
python backend/models_training/audio_model/audiocleaning.py \
  --input uncleaned_datasets/cleaned_data/classified_audiofiles \
  --out output_weekwise/week1output/audio --sample 50
```

## Environment & instructor checklist
- Ensure `requirements.txt` includes `tensorflow` if `rnnlstm_archeitecture.py` uses Keras.
- Standardize dataset paths or add a small mapping helper for audio folder names.
- Pre-download torchvision pretrained weights (or allow internet access during Week 2).
- Prepare a teaching subset (5–50 samples per class) for CPU demos.
- Create `label_map.json` mapping audio fine-grained labels to the 4 target emotions.

## Deliverables summary
- Week 1: Preprocessed image subset + EDA.
- Week 2: Trained vision model + evaluation report + inference demo.
- Week 3: Audio `features.csv`, cleaned audio subset, EDA plots.
- Week 4: Trained audio model, feature selection report, integrated demo, final report.

## Grading rubric (suggested short)
- Functionality (30%): scripts run and produce expected outputs.
- Model training & metrics (30%): working models and evaluation.
- Analysis (20%): EDA, confusion-matrix interpretation, feature selection reasoning.
- Demo & report (20%): integrated demo and clear documentation.

---

If you'd like, I can now:
- Add slide notes / instructor talking points per day, or
- Create a small teaching subset generator script and a `label_map.json` file.
