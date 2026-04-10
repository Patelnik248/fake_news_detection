# Fake News Detection Pipeline 🛡️

A comprehensive, end-to-end Machine Learning pipeline for classifying news articles as Real or Fake. This project features dynamic model management via **MLflow**, a modern **Flask** web interface, and full **CI/CD** automation with **Jenkins** and **Docker**.

---

## 🚀 Key Features

*   **Dynamic Model Loading**: The API automatically fetches the "Best" performing model from the MLflow Tracking Server at runtime.
*   **Modern Web UI**: A clean, responsive interface for article analysis with real-time character counting and confidence scoring.
*   **Robust CI/CD**: Fully automated Build-Test-Deploy pipeline using Jenkins.
*   **Monitoring Stack**: Real-time system metrics via Prometheus and visual dashboards in Grafana.
*   **Nginx Proxy**: production-ready reverse proxying for the Flask application.

---

## 🛠️ Technology Stack

*   **Logic**: Python (Scikit-Learn, Pandas, NumPy)
*   **API/UI**: Flask, Waitress, Vanilla JS/CSS
*   **Tracking**: MLflow
*   **Orchestration**: Docker & Docker Compose
*   **CI/CD**: Jenkins
*   **Monitoring**: Prometheus & Grafana

---

## 📋 Prerequisites

*   **Windows 10/11**
*   **Anaconda/Miniconda** (for Python environment)
*   **Docker Desktop** (with WSL 2 backend)
*   **Jenkins** (Native Windows installation)

---

## 🏃 Getting Started (Local Run)

### 1. MLflow Tracking Server
Start the tracking server to manage model experiments:
```powershell
mlflow server --port 5000
```
Visit: `http://localhost:5000`

### 2. Training
Train the model and log it to MLflow:
```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
python src/train.py
```

### 3. Native API Execution
Run the API directly on your machine:
```powershell
$env:PORT=5001
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
python app/app.py
```
Visit: `http://localhost:5001`

---

## 🐳 Docker Deployment

To launch the full stack (API, Nginx, Monitoring) in one command:
```powershell
docker compose up -d --build
```

### System Architecture
| Service | URL | Description |
| :--- | :--- | :--- |
| **Frontend/Proxy** | `http://localhost:80` | Main User Interface via Nginx |
| **API Backend** | `http://localhost:5001` | Flask service (Waitress) |
| **MLflow UI** | `http://localhost:5000` | Experiment & Model Management |
| **Prometheus** | `http://localhost:9090` | System Metric Collection |
| **Grafana** | `http://localhost:3000` | Analytics Dashboards |

---

## 🔄 Jenkins CI/CD Pipeline

The project includes a `Jenkinsfile` optimized for Windows. 

### Pipeline Stages:
1.  **Checkout**: Pulls code from GitHub.
2.  **Test**: Installs dependencies and runs `pytest`.
3.  **Train Model**: Automatically retrains if data or model code changes.
4.  **Build**: Creates a versioned Docker image.
5.  **Deploy**: Atomically cleans old containers and deploys the new stack via Compose.
6.  **Health Check**: Verifies the API is up and the MLflow model is loaded.

---

## 📊 API Endpoints

*   `GET /`: Serves the Web UI.
*   `POST /predict`: Takes `{"text": "..."}` and returns the classification.
*   `GET /health`: Returns service health status.
*   `GET /metrics`: Exposes Prometheus metrics.

---

## 🤝 Contributing
1.  Push changes to GitHub.
2.  Jenkins will automatically trigger a build and deploy the update if all tests pass.
