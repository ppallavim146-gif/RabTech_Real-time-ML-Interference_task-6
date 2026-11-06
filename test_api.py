from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_schema():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()
    assert "vectorizer_loaded" in response.json()


def test_prediction_schema():
    response = client.post("/predict", json={"text": "I really enjoyed this movie!"})
    assert response.status_code == 200
    data = response.json()
    assert {"text", "predicted_sentiment", "decision_score", "score_type"} <= data.keys()
    assert data["predicted_sentiment"] in ["Positive", "Negative"]
    assert isinstance(data["decision_score"], (int, float))


def test_negative_review():
    response = client.post("/predict", json={"text": "This movie was awful and boring."})
    assert response.status_code == 200
    assert response.json()["predicted_sentiment"] == "Negative"


def test_empty_text():
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_whitespace_text():
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_missing_text():
    assert client.post("/predict", json={}).status_code == 422


def test_text_too_long():
    assert client.post("/predict", json={"text": "x" * 5001}).status_code == 422
