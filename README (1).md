# Real-Time ML Inference REST API & Capstone

A FastAPI microservice for movie-review sentiment classification. It loads a trained scikit-learn LinearSVC model and its fitted TF-IDF vectorizer, validates JSON input, and returns a sentiment label with a decision score.

> **Score clarification:** LinearSVC does not natively provide calibrated probabilities. This API returns `decision_score`, not probability. A positive score indicates the positive class; a negative score indicates the negative class.

## Architecture

```text
Client -> POST /predict (JSON text)
       -> FastAPI/Pydantic validation
       -> Fitted TF-IDF vectorizer (review -> sparse features)
       -> LinearSVC (class + decision score)
       -> JSON response
```

## Model and evaluation

The model artifacts were trained separately on IMDb movie reviews. The training workflow compared Logistic Regression, Multinomial Naive Bayes, and LinearSVC and selected LinearSVC.

| Candidate | Reported test accuracy |
|---|---:|
| Logistic Regression | 87.15% |
| Multinomial Naive Bayes | 84.65% |
| LinearSVC (champion) | 86.75% |

The fitted TF-IDF vectorizer uses up to 20,000 features and unigram/bigram features. Inference uses serialized artifacts; the API does not train on startup.

## Required artifacts

Ensure these trained files are present in `artifacts/`:

```text
artifacts/
├── sentiment_model.joblib
└── tfidf_vectorizer.joblib
```

The source bundle does not include these binary files; copy the actual trained artifacts from your working project.

## Run locally

Use Python 3.11 (same base version as the container).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Docs: <http://127.0.0.1:8000/docs>  
Health: <http://127.0.0.1:8000/health>

## API

### `GET /`
Returns service status and docs path.

### `GET /health`
Reports whether model and vectorizer loaded.

### `POST /predict`
Request:

```json
{"text": "I really enjoyed this movie!"}
```

Example response (score is illustrative):

```json
{
  "text": "I really enjoyed this movie!",
  "predicted_sentiment": "Positive",
  "decision_score": 0.83,
  "score_type": "linear_svc_decision_score"
}
```

Validation and errors:
- `422`: missing, blank, invalid, or longer-than-5,000-character text.
- `503`: model/vectorizer artifacts unavailable.
- `500`: unexpected inference error.

## Tests

```powershell
python -m pytest -v
```

The suite checks status endpoints, prediction schema and behavior, and invalid request response codes. Prediction tests require the artifacts.

## Docker

Build from the repository root:

```powershell
docker build -t real-time-ml-inference-api .
docker run --rm -p 8002:8000 real-time-ml-inference-api
```

Open <http://127.0.0.1:8002/docs> or <http://127.0.0.1:8002/health>.

## End-to-end workflow

1. Prepare and clean review text.
2. Split data into training and evaluation sets.
3. Fit TF-IDF on training data.
4. Train and evaluate candidate classifiers.
5. Select and serialize the champion model and fitted vectorizer.
6. Load artifacts in the FastAPI service.
7. Validate the incoming JSON payload.
8. Transform text and perform inference.
9. Return sentiment and decision score.
10. Test and containerize the API.

## Limitations and future improvements

- LinearSVC decision scores are not probabilities.
- Probability output requires a calibrated classifier or a model supporting `predict_proba`, followed by evaluation.
- Add authentication, rate limiting, structured logs, monitoring, and deployment hardening before public production use.

## Certification checklist

- [ ] Put both model artifacts in `artifacts/`.
- [ ] Run `python -m pytest -v`.
- [ ] Start API and verify `/docs`.
- [ ] Build and run Docker image.
- [ ] Push source, tests, Dockerfile, pinned dependencies, artifacts, and README to GitHub.
- [ ] Submit the GitHub repository URL.
