import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from utils import preprocess_image, extract_audio_features

app = Flask(__name__)

import joblib

# ── Path Setup ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # This is /backend/
ROOT_DIR = os.path.dirname(BASE_DIR)                   # This is /ai_sentiment/

# Search paths for models
AUDIO_MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'cat_emotion_rfc.pkl')
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'cat_emotion_cnn.keras')

# Initialize models
audio_model   = None
image_model   = None
MODELS_LOADED = False

try:
    if os.path.exists(AUDIO_MODEL_PATH):
        print(f"Loading Audio Model from: {AUDIO_MODEL_PATH}")
        audio_model = joblib.load(AUDIO_MODEL_PATH)
    else:
        print(f"Warning: Audio model not found at {AUDIO_MODEL_PATH}")

    if os.path.exists(IMAGE_MODEL_PATH):
        print(f"Loading Image Model from: {IMAGE_MODEL_PATH}")
        image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    else:
        print(f"Warning: Image model not found at {IMAGE_MODEL_PATH}")

    if audio_model is not None or image_model is not None:
        MODELS_LOADED = True
        print("Success: At least one model loaded successfully.")

except Exception as e:
    print(f"Critical Error: Failed to load models. Error: {e}")
    MODELS_LOADED = False

CLASSES = ['angry', 'fearful', 'happy', 'sad']

EMOTION_TIPS = {
    'happy':   "Your cat is happy! Keep up the playtime and affection. 😺",
    'sad':     "Your cat seems sad. Try gentle cuddles, extra playtime, or check for any discomfort. 😿",
    'angry':   "Your cat is angry. Give them space, avoid sudden movements, and check for stressors. 😾",
    'fearful': "Your cat is fearful. Create a safe hiding spot, speak softly, and remove the source of fear. 🙀"
}

def save_upload(file):
    os.makedirs('uploads', exist_ok=True)
    path = os.path.join('uploads', file.filename)
    file.save(path)
    return path

def run_prediction(model, features, mock=False):
    """
    Safely runs prediction. 
    If model is None or MODELS_LOADED is False, returns mock results.
    Works for both Keras (predict) and Scikit-Learn (predict_proba).
    """
    if mock or not MODELS_LOADED or model is None:
        # Mock prediction (random distribution)
        pred = np.random.dirichlet(np.ones(4), size=1)[0]
    else:
        try:
            # Check if it's a Scikit-Learn model (has predict_proba)
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(features)[0]
            else:
                # Keras model
                pred = model.predict(features)[0]
        except Exception as e:
            print(f"Prediction Error: {e}")
            pred = np.random.dirichlet(np.ones(4), size=1)[0]
            
    idx = int(np.argmax(pred))
    return pred, idx

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file_path = save_upload(request.files['file'])
    try:
        features = extract_audio_features(file_path)
        pred, idx = run_prediction(audio_model, features)
        all_scores = {CLASSES[i]: float(pred[i]) for i in range(len(CLASSES))}
        return jsonify({
            "prediction": CLASSES[idx],
            "confidence": round(float(pred[idx]) * 100, 2),
            "all_scores": all_scores,
            "tip": EMOTION_TIPS[CLASSES[idx]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file_path = save_upload(request.files['file'])
    try:
        features = preprocess_image(file_path)
        pred, idx = run_prediction(image_model, features)
        all_scores = {CLASSES[i]: float(pred[i]) for i in range(len(CLASSES))}
        return jsonify({
            "prediction": CLASSES[idx],
            "confidence": round(float(pred[idx]) * 100, 2),
            "all_scores": all_scores,
            "tip": EMOTION_TIPS[CLASSES[idx]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_fusion', methods=['POST'])
def predict_fusion():
    """Combine image + audio predictions via weighted average."""
    if 'image' not in request.files or 'audio' not in request.files:
        return jsonify({"error": "Both image and audio files are required"}), 400

    img_path   = save_upload(request.files['image'])
    audio_path = save_upload(request.files['audio'])

    try:
        img_features   = preprocess_image(img_path)
        audio_features = extract_audio_features(audio_path)

        img_pred, _   = run_prediction(image_model, img_features)
        audio_pred, _ = run_prediction(audio_model, audio_features)

        # Weighted fusion: image 60%, audio 40%
        fused = 0.6 * img_pred + 0.4 * audio_pred
        idx   = int(np.argmax(fused))
        all_scores = {CLASSES[i]: float(fused[i]) for i in range(len(CLASSES))}

        return jsonify({
            "prediction": CLASSES[idx],
            "confidence": round(float(fused[idx]) * 100, 2),
            "all_scores": all_scores,
            "tip": EMOTION_TIPS[CLASSES[idx]],
            "image_prediction":  CLASSES[int(np.argmax(img_pred))],
            "audio_prediction":  CLASSES[int(np.argmax(audio_pred))]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
