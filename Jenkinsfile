pipeline {

    // Run on any available Jenkins agent
    agent any

    // ── Environment variables ─────────────────────────────────
    environment {
        IMAGE_NAME    = "fake-news-api"
        IMAGE_TAG     = "${BUILD_NUMBER}"
        COMPOSE_FILE  = "docker-compose.yml"
    }

    // ── Pipeline Stages ───────────────────────────────────────
    stages {

        // Stage 1: Pull latest code from GitHub
        stage("Checkout") {
            steps {
                echo "==> Pulling latest code from GitHub..."
                checkout scm
            }
        }

        // Stage 2: Install Python deps and run tests
        stage("Test") {
            steps {
                echo "==> Installing dependencies and running tests..."
                bat """
                    pip install --quiet -r requirements.txt
                    python -m pytest tests/ -v --tb=short
                """
            }
        }

        // Stage 3: (Optional) Train model — only when data changes
        stage("Train Model") {
            when {
                anyOf {
                    changeset "data/**"        // retrain if data changes
                    changeset "src/model.py"   // or if model code changes
                    expression { params.FORCE_TRAIN == true }
                }
            }
            steps {
                echo "==> Training model..."
                bat """
                    set MLFLOW_TRACKING_URI=http://localhost:5000
                    python src/train.py
                """
            }
        }

        // Stage 4: Build Docker image
        stage("Build Docker Image") {
            steps {
                echo "==> Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}..."
                bat "docker build -t %IMAGE_NAME%:%IMAGE_TAG% -t %IMAGE_NAME%:latest ."
            }
        }

        // Stage 5: Deploy with docker-compose
        stage("Deploy") {
            steps {
                echo "==> Deploying flask-api and nginx..."
                bat """
                    docker compose -p fake-news-detection -f ${COMPOSE_FILE} up -d --build flask-api nginx
                """
            }
        }

        // Stage 6: Verify deployment is healthy
        stage("Health Check") {
            steps {
                echo "==> Waiting for API to start and load model..."
                // Using PowerShell for a more reliable sleep and curl alternative on Windows
                powershell """
                    Start-Sleep -Seconds 20
                    Invoke-WebRequest -Uri "http://localhost:5001/health" -UseBasicParsing
                    Write-Host "API is healthy!"
                """
            }
        }
    }

    // ── Post-build actions ────────────────────────────────────
    post {
        success {
            echo "Pipeline PASSED. App deployed successfully."
        }
        failure {
            echo "Pipeline FAILED. Check logs above."
        }
        always {
            echo "Cleaning up unused Docker images..."
            bat "docker image prune -f || true"
        }
    }
}
