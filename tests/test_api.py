import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock
from app.app import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── Test: Health check ────────────────────────────────────────
def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"


# ── Test: API Info route ─────────────────────────────────────
def test_api_info(client):
    res = client.get("/api/info")
    assert res.status_code == 200
    assert "Fake News" in res.get_json()["service"]


# ── Test: Missing text field ──────────────────────────────────
def test_predict_missing_text(client):
    res = client.post("/predict", json={"wrong_field": "hello"})
    assert res.status_code == 400
    assert "error" in res.get_json()


# ── Test: Text too short ──────────────────────────────────────
def test_predict_short_text(client):
    res = client.post("/predict", json={"text": "hi"})
    assert res.status_code == 400


# ── Test: Valid prediction (mock model) ───────────────────────
def test_predict_valid(client):
    import numpy as np
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]                       # Real
    mock_model.predict_proba.return_value = [[0.05, 0.95]]      # 95% confident

    with patch("app.app.get_model", return_value=mock_model):
        res = client.post("/predict", json={
            "text": "Scientists have confirmed a major breakthrough in cancer research."
        })
        assert res.status_code == 200
        data = res.get_json()
        assert data["label"] == "Real"
        assert data["confidence"] > 0.9
        assert data["is_fake"] is False


# ── Test: Fake prediction ─────────────────────────────────────
def test_predict_fake(client):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]                       # Fake
    mock_model.predict_proba.return_value = [[0.88, 0.12]]

    with patch("app.app.get_model", return_value=mock_model):
        res = client.post("/predict", json={
            "text": "Aliens have secretly taken over the White House says anonymous source."
        })
        data = res.get_json()
        assert data["label"] == "Fake"
        assert data["is_fake"] is True


# ── Test: Metrics endpoint ─────────────────────────────────────
def test_metrics_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert b"api_request_count_total" in res.data
