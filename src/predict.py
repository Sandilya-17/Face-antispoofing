"""
Run inference on a single image using the trained pipeline.

Usage:
    python predict.py path/to/image.jpg
"""
import sys
import os
import joblib
import numpy as np

from feature_extraction import extract_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "models")


def load_pipeline():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    return scaler, pca, model


def predict(path):
    scaler, pca, model = load_pipeline()
    feats = extract_features(path).reshape(1, -1)
    feats_s = scaler.transform(feats)
    feats_p = pca.transform(feats_s)
    pred = model.predict(feats_p)[0]
    prob = model.predict_proba(feats_p)[0, 1]
    label = "FAKE (spoof)" if pred == 1 else "REAL"
    print(f"{path}: {label}  (spoof probability = {prob:.3f})")
    return label, prob


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
