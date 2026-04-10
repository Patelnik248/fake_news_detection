# =============================================================
# train.py — Member 1: Training Pipeline + MLflow Tracking
# Run with: python src/train.py
# =============================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mlflow
import mlflow.sklearn
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)

from src.preprocess import get_dataset
from src.model import build_model, save_model

# ── MLflow tracking server URI ────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "fake-news-detection"

def evaluate(model, X_test, y_test):
    """Compute and print evaluation metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }
    print("\n" + classification_report(y_test, y_pred,
          target_names=["Fake", "Real"]))
    return metrics


def train():
    """Main training function with MLflow experiment tracking."""
    # ── 1. Set up MLflow ──────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── 2. Load and preprocess data ───────────────────────────
    print("[STEP 1] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = get_dataset()

    # ── 3. Build model ────────────────────────────────────────
    print("\n[STEP 2] Building model pipeline...")
    model = build_model()

    with mlflow.start_run():
        # ── 4. Log hyperparameters ────────────────────────────
        mlflow.log_param("model_type",   "LogisticRegression")
        mlflow.log_param("tfidf_max_features", 50000)
        mlflow.log_param("ngram_range",  "(1, 2)")
        mlflow.log_param("C",            1.0)
        mlflow.log_param("train_size",   len(X_train))
        mlflow.log_param("test_size",    len(X_test))

        # ── 5. Train ──────────────────────────────────────────
        print("\n[STEP 3] Training model...")
        model.fit(X_train, y_train)

        # ── 6. Evaluate ───────────────────────────────────────
        print("\n[STEP 4] Evaluating model...")
        metrics = evaluate(model, X_test, y_test)

        # ── 7. Log metrics to MLflow ──────────────────────────
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            print(f"  {name}: {value:.4f}")

        # ── 8. Log model artifact ─────────────────────────────
        mlflow.sklearn.log_model(model, "model")
        print("\n[INFO] MLflow run complete.")

    # ── 9. Save model locally ─────────────────────────────────
    print("\n[STEP 5] Saving model...")
    save_model(model)
    print("\n[DONE] Training complete!")


if __name__ == "__main__":
    train()
