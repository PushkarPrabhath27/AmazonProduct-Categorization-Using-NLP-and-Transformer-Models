"""FastAPI application for product categorization inference."""

from pathlib import Path
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import inference functions
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from inference import BaselinePredictor, BERTPredictor

app = FastAPI(title="Amazon Product Categorization API", version="1.0.0")

# Initialize predictors (lazy loading)
_baseline_predictor = None
_bert_predictor = None


class PredictionRequest(BaseModel):
    title: str
    description: str = ""
    top_k: int = 3
    model_type: str = "baseline"  # "baseline" or "bert"


class PredictionResponse(BaseModel):
    predictions: List[Tuple[str, float]]
    model_type: str


def get_baseline_predictor():
    """Lazy load baseline predictor."""
    global _baseline_predictor
    if _baseline_predictor is None:
        from pathlib import Path

        MODEL_DIR = PROJECT_ROOT / "models"
        _baseline_predictor = BaselinePredictor(
            MODEL_DIR / "baseline.joblib",
            MODEL_DIR / "tfidf_vectorizer.joblib",
            MODEL_DIR / "label_encoder.joblib",
        )
    return _baseline_predictor


def get_bert_predictor():
    """Lazy load BERT predictor."""
    global _bert_predictor
    if _bert_predictor is None:
        BERT_MODEL_DIR = PROJECT_ROOT / "models" / "bert_final"
        _bert_predictor = BERTPredictor(BERT_MODEL_DIR, device="cpu")
    return _bert_predictor


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Amazon Product Categorization API", "version": "1.0.0"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict product category."""
    try:
        if request.model_type == "baseline":
            predictor = get_baseline_predictor()
        elif request.model_type == "bert":
            predictor = get_bert_predictor()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model_type: {request.model_type}")

        predictions = predictor.predict(request.title, request.description, top_k=request.top_k)

        return PredictionResponse(predictions=predictions, model_type=request.model_type)

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

