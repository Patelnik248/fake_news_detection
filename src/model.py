# =============================================================
# model.py — Member 1: Model Definition
# TF-IDF + Logistic Regression pipeline
# =============================================================

import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "models/fake_news_model.pkl"
os.makedirs("models", exist_ok=True)

def build_model():
    """
    Build a simple sklearn Pipeline:
      Step 1: TF-IDF — converts text to numerical feature vectors
      Step 2: Logistic Regression — binary classification (real vs fake)
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,    # use top 50k words
            ngram_range=(1, 2),    # unigrams and bigrams
            stop_words="english"   # remove common English words
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,                 # regularization strength
            solver="lbfgs",
            n_jobs=-1              # use all CPU cores
        ))
    ])
    return pipeline


def save_model(model, path=MODEL_PATH):
    """Save trained model to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved to {path}")


def load_model(path=MODEL_PATH):
    """Load a saved model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}. Train first.")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Model loaded from {path}")
    return model


def predict(model, texts):
    """
    Run prediction on a list of text strings.
    Returns: labels (0=Fake, 1=Real), probabilities
    """
    labels = model.predict(texts)
    probs  = model.predict_proba(texts)
    return labels, probs
