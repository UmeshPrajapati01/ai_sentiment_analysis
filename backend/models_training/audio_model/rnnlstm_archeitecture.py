# ...existing code...
"""
Audio model tasks B5-B8:
- load_features(csv_path)
- select_features(X, y, k)
- build_lstm(input_shape, n_classes)
- train_model(X, y, output_dir, params)
- evaluate_model(model, X_test, y_test, label_encoder, output_dir)

Expect features CSV (from audiocleaning) with columns:
  filename,label,duration,<feature columns...>

Default input (features.csv) -> uncleaned_datasets/.../week1output/audio/features.csv
Default output -> output_weekwise/week2output/audio
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import tensorflow as tf
from tensorflow import keras

Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint
def load_features(csv_path: str):
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column")
    X = df.drop(columns=['filename','label','duration'], errors='ignore')
    y = df['label'].astype(str)
    return X, y

def select_features_rf(X: pd.DataFrame, y: pd.Series, k: int = 30):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X.fillna(0), y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected = importances.iloc[:min(k, len(importances))].index.tolist()
    return selected, importances

def build_lstm(input_shape, n_classes):
    model = Sequential([
        LSTM(128, input_shape=input_shape, activation='tanh'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X, y, output_dir: str, params: dict):
    os.makedirs(output_dir, exist_ok=True)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    # scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # reshape for LSTM: (samples, timesteps=1, features)
    Xs = Xs.reshape((Xs.shape[0], 1, Xs.shape[1]))
    # split
    X_train, X_tmp, y_train, y_tmp = train_test_split(Xs, y_enc, test_size=0.3, stratify=y_enc, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    # to_categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
    model = build_lstm(input_shape=(Xs.shape[1], Xs.shape[2]), n_classes=num_classes)
    # callbacks
    ckpt_path = os.path.join(output_dir, "best_model.h5")
    cb = [EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
          ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True)]
    hist = model.fit(X_train, y_train_cat, validation_data=(X_val, y_val_cat),
                     epochs=params.get('epochs',50), batch_size=params.get('batch_size',32),
                     callbacks=cb, verbose=2)
    # save artifacts
    # primary save location (week2 output)
    final_out = os.path.join(output_dir, "final_model.h5")
    model.save(final_out)
    with open(os.path.join(output_dir, "label_encoder.json"), "w", encoding="utf8") as f:
        json.dump(le.classes_.tolist(), f)
    pd.to_pickle(scaler, os.path.join(output_dir, "scaler.pkl"))
    # additional central trained-model repo for backend usage
    trained_dir = os.path.join("backend", "trained_modelfiles", "audio_model")
    os.makedirs(trained_dir, exist_ok=True)
    model.save(os.path.join(trained_dir, "audio_model_final.h5"))
    # copy encoder and scaler to trained repo as well
    with open(os.path.join(trained_dir, "label_encoder.json"), "w", encoding="utf8") as f:
        json.dump(le.classes_.tolist(), f)
    pd.to_pickle(scaler, os.path.join(trained_dir, "scaler.pkl"))
    # save history plot
    plt.figure()
    plt.plot(hist.history['loss'], label='train_loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.legend(); plt.title("Loss")
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150); plt.close()
    plt.figure()
    plt.plot(hist.history.get('accuracy', hist.history.get('acc')), label='train_acc')
    plt.plot(hist.history.get('val_accuracy', hist.history.get('val_acc')), label='val_acc')
    plt.legend(); plt.title("Accuracy")
    plt.savefig(os.path.join(output_dir, "acc_curve.png"), dpi=150); plt.close()
    return model, (X_test, y_test, y_test_cat), le, scaler

def evaluate_model(model, X_test_tuple, label_encoder, output_dir: str):
    X_test, y_test, y_test_cat = X_test_tuple
    os.makedirs(output_dir, exist_ok=True)
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    # classification report
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf8") as f:
        f.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap='Blues'); plt.colorbar()
    ticks = np.arange(len(label_encoder.classes_))
    plt.xticks(ticks, label_encoder.classes_, rotation=45, ha='right')
    plt.yticks(ticks, label_encoder.classes_)
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150); plt.close()
    # precision-recall curves (one-vs-rest)
    y_bin = label_binarize(y_test, classes=np.arange(len(label_encoder.classes_)))
    plt.figure(figsize=(8,6))
    for i, cls in enumerate(label_encoder.classes_):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{cls} (AUC={pr_auc:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves"); plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, "precision_recall.png"), dpi=150); plt.close()

def select_and_train_pipeline(features_csv: str, output_dir: str, top_k: int = 40, params: dict = None):
    params = params or {}
    X_df, y = load_features(features_csv)
    selected, importances = select_features_rf(X_df, y, k=top_k)
    os.makedirs(output_dir, exist_ok=True)
    importances.to_csv(os.path.join(output_dir, "feature_importances.csv"))
    X_sel = X_df[selected].fillna(0)
    model, test_tuple, le, scaler = train_model(X_sel, y, output_dir, params)
    evaluate_model(model, test_tuple, le, output_dir)
    print("Training + evaluation complete. Artifacts in", output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="B5-B8 audio model training pipeline")
    parser.add_argument("--features-csv", type=str,
                        default=r"C:\Users\ramak\OneDrive\Desktop\project_smartx\project_mew\Dev-of-an-AI-Based-Cat-Emotion-Recognition-System-Using-Facial-and-Vocal-Analysis_Feb_Batch-8_2026\output_weekwise\week1output\audio\features.csv",
                        help="Input features CSV from audio cleaning step")
    parser.add_argument("--output-dir", type=str,
                        default=r"C:\Users\ramak\OneDrive\Desktop\project_smartx\project_mew\Dev-of-an-AI-Based-Cat-Emotion-Recognition-System-Using-Facial-and-Vocal-Analysis_Feb_Batch-8_2026\output_weekwise\week2output\audio",
                        help="Output folder for models and plots")
    parser.add_argument("--top-k", type=int, default=40, help="Select top-K features")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    select_and_train_pipeline(args.features_csv, args.output_dir, top_k=args.top_k,
                              params={'epochs': args.epochs, 'batch_size': args.batch_size})