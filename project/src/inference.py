"""Inference script for product categorization.

Loads trained models and provides prediction interface for new product data.
"""

import os
import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

# Setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_baseline_model():
    """Load baseline model and preprocessing artifacts."""
    logging.info("Loading baseline model...")
    
    model = joblib.load(MODEL_DIR / "baseline.joblib")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    le = joblib.load(MODEL_DIR / "label_encoder.joblib")
    
    return model, vectorizer, le


def load_bert_model():
    """Load BERT model and tokenizer."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        logging.info("Loading BERT model...")
        
        model_path = MODEL_DIR / "bert_final"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        le = joblib.load(model_path / "label_encoder.joblib")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        return model, tokenizer, le, device
    except Exception as e:
        logging.warning(f"Could not load BERT model: {e}")
        return None, None, None, None


def preprocess_text(title, description):
    """Combine and preprocess title and description."""
    title = str(title) if title else ""
    description = str(description) if description else ""
    
    # Simple preprocessing (matching training)
    text = f"{title.strip()} {description.strip()}".lower()
    
    return text


def predict_baseline(title, description, top_k=3):
    """Predict using baseline model."""
    model, vectorizer, le = load_baseline_model()
    
    # Preprocess
    text = preprocess_text(title, description)
    
    # Vectorize
    X = vectorizer.transform([text])
    
    # Predict
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    
    # Get top-k predictions
    top_k_indices = np.argsort(proba)[-top_k:][::-1]
    
    results = []
    for idx in top_k_indices:
        results.append({
            'category': le.classes_[idx],
            'probability': float(proba[idx])
        })
    
    return {
        'predicted_category': le.classes_[pred],
        'confidence': float(proba[pred]),
        'top_k_predictions': results
    }


def predict_bert(title, description, top_k=3):
    """Predict using BERT model."""
    import torch
    
    model, tokenizer, le, device = load_bert_model()
    
    if model is None:
        raise ValueError("BERT model not available")
    
    # Preprocess
    text = preprocess_text(title, description)
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        proba = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    
    pred = np.argmax(proba)
    
    # Get top-k predictions
    top_k_indices = np.argsort(proba)[-top_k:][::-1]
    
    results = []
    for idx in top_k_indices:
        results.append({
            'category': le.classes_[idx],
            'probability': float(proba[idx])
        })
    
    return {
        'predicted_category': le.classes_[pred],
        'confidence': float(proba[pred]),
        'top_k_predictions': results
    }


def predict(title, description, model_type="baseline", top_k=3):
    """Main prediction function.
    
    Args:
        title: Product title
        description: Product description
        model_type: "baseline" or "bert"
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions and probabilities
    """
    if model_type == "baseline":
        return predict_baseline(title, description, top_k)
    elif model_type == "bert":
        return predict_bert(title, description, top_k)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict product category from title and description"
    )
    parser.add_argument("--title", type=str, required=True, help="Product title")
    parser.add_argument("--desc", "--description", type=str, default="", help="Product description")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "bert"],
                      help="Model to use for prediction")
    parser.add_argument("--top-k", "--top_k", type=int, default=3, help="Number of top predictions")
    
    args = parser.parse_args()
    
    print(f"\nProduct Title: {args.title}")
    print(f"Product Description: {args.desc}")
    print(f"Model: {args.model}")
    print("-" * 70)
    
    try:
        result = predict(args.title, args.desc, args.model, args.top_k)
        
        print(f"\nPredicted Category: {result['predicted_category']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nTop {args.top_k} Predictions:")
        
        for i, pred in enumerate(result['top_k_predictions'], 1):
            print(f"  {i}. {pred['category']:<30} ({pred['probability']:.2%})")
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
