import os
import sys
import time
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify, send_from_directory
from waitress import serve
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import mlflow
import mlflow.sklearn

from src.model import load_model, predict
from tracking.mlflow_setup import load_best_model

# ── Logging Setup ─────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log", mode="a")
    ]
)
logger = logging.getLogger("fake-news-api")

# ── Prometheus Metrics ────────────────────────────────────────
REQUEST_COUNT = Counter(
    "api_request_count_total",
    "Total number of API requests",
    ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "api_prediction_count_total",
    "Number of predictions made",
    ["result"]  # "real" or "fake"
)

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


# ── Flask App ─────────────────────────────────────────────────
app = Flask(__name__)

# Load model once at startup (not on every request)
model = None

def get_model():
    """Lazy-load the model — tries MLflow first, then Local File."""
    global model
    if model is None:
        # ── 1. Try MLflow first ────────────────────────────────
        try:
            logger.info("Connecting to MLflow Tracking Server...")
            model = load_best_model()
            logger.info("Successfully loaded best model from MLflow.")
        except Exception as e:
            logger.warning(f"MLflow fetch failed ({e}). Falling back to local file.")
            
            # ── 2. Fallback to local model.pkl ────────────────
            try:
                model = load_model()
                logger.info("Successfully loaded model from local disk.")
            except Exception as e_local:
                logger.error(f"Critical error: No model available. {e_local}")
                raise e_local
                
    return model


# ── Routes ────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint — used by Docker/load balancer."""
    return jsonify({"status": "ok", "service": "fake-news-api"}), 200


@app.route("/predict", methods=["POST"])
def make_prediction():
    """
    POST /predict
    Body: {"text": "some news article content"}
    Returns: {"label": "Real"/"Fake", "confidence": 0.95}
    """
    start_time = time.time()
    status_code = 200

    try:
        # ── Validate input ────────────────────────────────────
        data = request.get_json()
        if not data or "text" not in data:
            logger.warning("Missing 'text' field in request")
            REQUEST_COUNT.labels("/predict", "POST", "400").inc()
            return jsonify({"error": "Request body must include a 'text' field."}), 400

        text = data["text"].strip()
        if len(text) < 10:
            return jsonify({"error": "Text is too short to classify."}), 400

        # ── Run prediction ────────────────────────────────────
        clf = get_model()
        labels, probs = predict(clf, [text])

        label      = "Real" if labels[0] == 1 else "Fake"
        confidence = float(max(probs[0]))

        logger.info(f"Prediction: {label} (confidence={confidence:.3f}) | "
                    f"text_length={len(text)}")

        # Track prediction stats
        PREDICTION_COUNT.labels(label.lower()).inc()

        response = {
            "label":      label,
            "confidence": round(confidence, 4),
            "is_fake":    bool(labels[0] == 0)
        }

    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        status_code = 503
        response    = {"error": "Model not ready. Please train the model first."}

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        status_code = 500
        response    = {"error": "Internal server error."}

    # ── Track latency + request count ────────────────────────
    latency = time.time() - start_time
    REQUEST_LATENCY.labels("/predict").observe(latency)
    REQUEST_COUNT.labels("/predict", "POST", str(status_code)).inc()

    return jsonify(response), status_code


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus scrape endpoint — exposes app metrics."""
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/api/info", methods=["GET"])
def api_info():
    """Returns basic service information (moved from root)."""
    return jsonify({
        "service": "Fake News Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Classify news as Real or Fake",
            "GET  /health":  "Health check",
            "GET  /metrics": "Prometheus metrics",
        }
    })


@app.route("/", methods=["GET"])
def index():
    """Serves the modern frontend UI."""
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    """Serve static files from the root for better compatibility."""
    return send_from_directory(STATIC_DIR, path)


# ── Start Server ──────────────────────────────────────────────
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 5001))
    logger.info(f"Starting Waitress server on port {PORT}...")
    serve(app, host="0.0.0.0", port=PORT, threads=4)
