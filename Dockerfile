# ── Stage 1: Base image ───────────────────────────────────────
FROM python:3.11-slim AS base

# Set working directory inside the container
WORKDIR /app

# ── Stage 2: Install dependencies ────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 --retries=3 -r requirements.txt

# ── Stage 3: Copy application code ───────────────────────────
COPY src/         ./src/
COPY app/app.py   ./app/app.py
COPY app/static/  ./app/static/
COPY models/      ./models/

# Create logs directory
RUN mkdir -p logs

# ── Environment variables ─────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PORT=5001
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# ── Expose API port ───────────────────────────────────────────
EXPOSE 5001

# ── Health check (Docker will poll this) ─────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')"

# ── Start the Flask + Waitress server ────────────────────────
CMD ["python", "app/app.py"]
