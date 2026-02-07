# Dev-of-an-AI-Based-Cat-Emotion-Recognition-System-Using-Facial-and-Vocal-Analysis_Feb_Batch-8_2026
# MeowMood: AI-Based Cat Emotion Recognition System

**Project Title**  
Development of an AI-Based Cat Emotion Recognition System Using Facial and Vocal Analysis

**Final Year Project** – rama (Kukatpally, Telangana)

---

## 🎯 Project Overview & Problem Statement

Cat owners often misinterpret their pet's emotions because feline body language and vocalizations are subtle and complex.  
This project aims to build an **AI system** that analyzes **cat facial expressions** (images) and **meows/vocalizations** (audio) to detect four emotions:

- Happy 😺
- Sad 😿
- Angry 😾
- Fearful 😨

The system will eventually be a **Streamlit web app** (html css and jacascript alsow)here users can upload a photo or audio clip of their cat and get:
- Predicted emotion
- Confidence score (%)
- Short behavioral interpretation

**Technologies Used (as per specification)**  
- Python 3.9+
- Deep Learning: PyTorch (preferred over TensorFlow for this project)
- Image: torchvision + PIL
- Audio: torchaudio (MFCC, spectrogram — Librosa not used due to env constraints)
- Web App: Streamlit
- Others: NumPy, Pandas, Matplotlib, Seaborn

---

## 📊 Current Status (Initial Commit)

- Repository created
- Branch strategy set up:
  - `educationla branch` → stable / final working version
  - `development` → active integration branch (all work happens here by your mentor you can check the code to study)
- Module 1 (Data Collection, Preprocessing & Transformation) in progress
- Folder structure created as per project specification
- Preprocessing scripts written for images and audio
- EDA (class distribution plots) partially implemented
- Datasets not yet fully uploaded (large files → will be added via .gitignore rules or external links)

**Vision Tasks in progress (A1–A4)**  
- A1: Image dataset acquisition & organization  
- A2: Image preprocessing (resize 64×64, grayscale, normalization)  
- A3: Augmentation (rotation, flip, brightness)  
- A4: EDA – class balance visualization  

**Audio Tasks in progress (B1–B4)**  
- B1: Audio dataset acquisition & organization  
- B2: Feature extraction (MFCC + Mel Spectrogram)  
- B3: Basic cleaning (resample, silence trimming)  
- B4: EDA visuals (waveform, spectrogram, pitch/intensity)

---

## 📂 Project Structure (Current)
