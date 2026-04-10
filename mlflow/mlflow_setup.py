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
    uri = f"runs:/{best.info.run_id}/model"
    return mlflow.sklearn.load_model(uri)


if __name__ == "__main__":
    setup_mlflow()
    list_runs()
