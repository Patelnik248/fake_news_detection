# =============================================================
# mlflow_setup.py — MLflow Utilities
# =============================================================

import mlflow
import mlflow.sklearn
import os

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "fake-news-detection"


def setup_mlflow():
    """Connect to MLflow and set the experiment."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[MLflow] Connected to {MLFLOW_URI}")


def list_runs():
    """List all past runs sorted by F1 score (best first)."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        print("No experiment found. Run training first.")
        return

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1 DESC"]
    )
    print(f"\n{'Run ID':<32} {'F1':>6} {'Accuracy':>9}")
    print("-" * 52)
    for r in runs:
        m = r.data.metrics
        print(f"{r.info.run_id:<32} {m.get('f1',0):.4f}  {m.get('accuracy',0):.4f}")


def get_best_run():
    """Return the run with the highest F1 score."""
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        return None
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1
    )
    return runs[0] if runs else None


def load_best_model():
    """Load the sklearn model from the best MLflow run."""
    best = get_best_run()
    if not best:
        raise RuntimeError("No runs found. Train first.")
    
    # ── 1. Try URL scheme (standard MLflow) ─────────────────────
    uri = f"runs:/{best.info.run_id}/model"
    
    try:
        # We try loading with a very short timeout concept (manual check)
        return mlflow.sklearn.load_model(uri)
    except Exception as e:
        print(f"[MLflow] Remote load failed ({e}). Checking local artifact root...")
        
        # ── 2. Fallback: Manual Local Path Discovery ─────────────
        # This bypasses the MLflow Artifact Proxy which often hangs on local Windows setups
        run_id = best.info.run_id
        
        # Search in known local locations
        possible_paths = [
            f"mlartifacts/1/{run_id}/artifacts/model",
            f"mlartifacts/1/models/m-{run_id}",               # seen in user environment
            f"mlruns/1/{run_id}/artifacts/model",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"[MLflow] Found local artifact at: {path}")
                return mlflow.sklearn.load_model(path)
                
        raise RuntimeError(f"Could not load model for run {run_id} via MLflow or local search.")


if __name__ == "__main__":
    setup_mlflow()
    list_runs()
