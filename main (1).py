from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "sentiment_model.joblib"
VECTORIZER_PATH = ROOT / "artifacts" / "tfidf_vectorizer.joblib"

app = FastAPI(title="Real-Time ML Inference API", version="1.0.0",
              description="Movie-review sentiment classification using a trained LinearSVC.")

model = vectorizer = None
load_error = None
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    if model.n_features_in_ != len(vectorizer.get_feature_names_out()):
        raise ValueError("Model and vectorizer feature counts do not match.")
except Exception as exc:
    load_error = str(exc)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Text must not be blank.")
        return value


class PredictionResponse(BaseModel):
    text: str
    predicted_sentiment: Literal["Positive", "Negative"]
    decision_score: float
    score_type: Literal["linear_svc_decision_score"] = "linear_svc_decision_score"


@app.get("/")
def root():
    return {"message": "Real-Time ML Inference API is running", "docs": "/docs"}


@app.get("/health")
def health():
    ready = model is not None and vectorizer is not None
    return {"status": "healthy" if ready else "unhealthy",
            "model_loaded": model is not None, "vectorizer_loaded": vectorizer is not None,
            "error": load_error}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model artifacts are unavailable.")
    try:
        features = vectorizer.transform([request.text])
        label = int(model.predict(features)[0])
        score = float(model.decision_function(features)[0])
        return PredictionResponse(text=request.text,
                                 predicted_sentiment="Positive" if label == 1 else "Negative",
                                 decision_score=score)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
