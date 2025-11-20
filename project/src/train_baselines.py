"""Train baseline models for product categorization.

Implements:
1. Logistic Regression (TF-IDF) with GridSearchCV
2. Random Forest (TF-IDF) with hyperparameter tuning
3. Multinomial Naive Bayes
4. LSTM with embedding layer
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DIR = PROCESSED_DIR / "features"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "experiments" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "train_baselines.log"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def load_data() -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Load TF-IDF features and labels."""
    logging.info("Loading TF-IDF features from %s", FEATURE_DIR)
    X_train = sparse.load_npz(FEATURE_DIR / "tfidf_train.npz")
    X_val = sparse.load_npz(FEATURE_DIR / "tfidf_val.npz")

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")

    # Encode labels
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["category"])
    y_val = le.transform(val_df["category"])

    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
    logging.info("Saved label encoder with %d classes", len(le.classes_))

    return X_train, X_val, y_train, y_val, le


def train_logistic_regression(
    X_train: sparse.csr_matrix, y_train: np.ndarray, X_val: sparse.csr_matrix, y_val: np.ndarray
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """Train Logistic Regression with GridSearchCV."""
    logging.info("Training Logistic Regression with GridSearchCV")
    param_grid = {"C": [0.01, 0.1, 1, 10], "max_iter": [1000]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, solver="liblinear"),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,  # More verbose output
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_precision": precision_score(y_val, y_pred, average="macro"),
        "macro_recall": recall_score(y_val, y_pred, average="macro"),
        "macro_f1": f1_score(y_val, y_pred, average="macro"),
    }

    logging.info("Best C: %s, Val macro-F1: %.4f", grid_search.best_params_["C"], metrics["macro_f1"])
    return best_model, metrics


def train_random_forest(
    X_train: sparse.csr_matrix, y_train: np.ndarray, X_val: sparse.csr_matrix, y_val: np.ndarray
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Train Random Forest with hyperparameter tuning."""
    logging.info("Training Random Forest - using subsample for faster training")
    
    # Subsample training data to make Random Forest training faster (80K is too many for GridSearchCV)
    from sklearn.model_selection import train_test_split
    _, X_train_sub, _, y_train_sub = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
    )
    logging.info("Using %d samples (25%% of training data) for Random Forest hyperparameter tuning", X_train_sub.shape[0])
    
    # Reduced parameter grid for faster training
    param_grid = {"n_estimators": [100], "max_depth": [20, 50]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold instead of 5

    logging.info("Starting GridSearchCV with %d parameter combinations", len(param_grid["n_estimators"]) * len(param_grid["max_depth"]))
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,  # Avoid nested parallelism issues
        verbose=2,  # More verbose output
    )
    grid_search.fit(X_train_sub, y_train_sub)
    
    logging.info("GridSearchCV completed, retraining on full training set with best params")
    # Retrain on full training data with best params
    best_model = RandomForestClassifier(**grid_search.best_params_, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = best_model.predict(X_val)


    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_precision": precision_score(y_val, y_pred, average="macro"),
        "macro_recall": recall_score(y_val, y_pred, average="macro"),
        "macro_f1": f1_score(y_val, y_pred, average="macro"),
    }

    logging.info("Best params: %s, Val macro-F1: %.4f", grid_search.best_params_, metrics["macro_f1"])
    return best_model, metrics


def train_naive_bayes(
    X_train: sparse.csr_matrix, y_train: np.ndarray, X_val: sparse.csr_matrix, y_val: np.ndarray
) -> Tuple[MultinomialNB, Dict[str, float]]:
    """Train Multinomial Naive Bayes."""
    logging.info("Training Multinomial Naive Bayes")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_precision": precision_score(y_val, y_pred, average="macro"),
        "macro_recall": recall_score(y_val, y_pred, average="macro"),
        "macro_f1": f1_score(y_val, y_pred, average="macro"),
    }

    logging.info("Val macro-F1: %.4f", metrics["macro_f1"])
    return model, metrics


class SimpleLSTM(nn.Module):
    """Simple LSTM for text classification."""

    def __init__(self, vocab_size: int, embed_dim: int = 100, hidden_dim: int = 128, num_classes: int = 15, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=dropout if dropout > 0 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


def train_lstm(
    X_train: sparse.csr_matrix,
    y_train: np.ndarray,
    X_val: sparse.csr_matrix,
    y_val: np.ndarray,
    num_classes: int,
    device: str = "cpu",
) -> Tuple[SimpleLSTM, Dict[str, float]]:
    """Train LSTM model (simplified - using TF-IDF indices as tokens)."""
    logging.info("Training LSTM model (simplified approach)")
    logging.warning("LSTM implementation uses TF-IDF feature indices as tokens (simplified)")

    # Convert sparse matrices to dense for LSTM (this is memory-intensive, consider subsampling)
    # For a proper LSTM, we'd need tokenized sequences, but for baseline we'll use a simplified approach
    max_features = X_train.shape[1]
    vocab_size = min(max_features, 10000)  # Limit vocab size

    # Use top features as vocabulary
    feature_sums = np.array(X_train.sum(axis=0)).flatten()
    top_features = np.argsort(feature_sums)[-vocab_size:]

    # Create simplified sequences (indices of non-zero features)
    def create_sequences(X: sparse.csr_matrix, top_features: np.ndarray, max_len: int = 100) -> np.ndarray:
        sequences = []
        for i in range(X.shape[0]):
            row = X[i].toarray().flatten()
            nonzero_indices = np.where(row > 0)[0]
            # Map to vocab indices
            vocab_indices = [np.where(top_features == idx)[0][0] for idx in nonzero_indices if idx in top_features]
            if len(vocab_indices) > max_len:
                vocab_indices = vocab_indices[:max_len]
            elif len(vocab_indices) < max_len:
                vocab_indices = vocab_indices + [0] * (max_len - len(vocab_indices))
            sequences.append(vocab_indices)
        return np.array(sequences)

    logging.info("Creating sequences for LSTM (this may take a while)")
    X_train_seq = create_sequences(X_train[:10000], top_features)  # Subsample for memory
    X_val_seq = create_sequences(X_val[:2000], top_features)
    y_train_sub = y_train[:10000]
    y_val_sub = y_val[:2000]

    model = SimpleLSTM(vocab_size=vocab_size, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    batch_size = 32
    num_epochs = 10
    best_val_f1 = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i in range(0, len(X_train_seq), batch_size):
            batch_X = torch.LongTensor(X_train_seq[i : i + batch_size]).to(device)
            batch_y = torch.LongTensor(y_train_sub[i : i + batch_size]).to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for i in range(0, len(X_val_seq), batch_size):
                batch_X = torch.LongTensor(X_val_seq[i : i + batch_size]).to(device)
                outputs = model(batch_X)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        val_f1 = f1_score(y_val_sub, val_preds, average="macro")
        logging.info("Epoch %d: Train loss: %.4f, Val macro-F1: %.4f", epoch + 1, train_loss / len(X_train_seq), val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / "lstm_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping at epoch %d", epoch + 1)
                break

    # Load best model
    model.load_state_dict(torch.load(MODEL_DIR / "lstm_best.pth"))
    model.eval()
    val_preds = []
    with torch.no_grad():
        for i in range(0, len(X_val_seq), batch_size):
            batch_X = torch.LongTensor(X_val_seq[i : i + batch_size]).to(device)
            outputs = model(batch_X)
            val_preds.extend(outputs.argmax(dim=1).cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(y_val_sub, val_preds),
        "macro_precision": precision_score(y_val_sub, val_preds, average="macro"),
        "macro_recall": recall_score(y_val_sub, val_preds, average="macro"),
        "macro_f1": f1_score(y_val_sub, val_preds, average="macro"),
    }

    return model, metrics


def save_results(all_metrics: Dict[str, Dict[str, float]], le) -> None:
    """Save metrics to CSV."""
    rows = []
    for model_name, metrics in all_metrics.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "metrics_baselines.csv"
    df.to_csv(output_path, index=False)
    logging.info("Saved baseline metrics to %s", output_path)

    # Also save per-class metrics for best model
    best_model_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]["macro_f1"])
    logging.info("Best baseline model: %s (macro-F1: %.4f)", best_model_name, all_metrics[best_model_name]["macro_f1"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training (memory-intensive)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for LSTM (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    X_train, X_val, y_train, y_val, le = load_data()
    num_classes = len(le.classes_)

    all_metrics = {}

    # 1. Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    joblib.dump(lr_model, MODEL_DIR / "baseline_lr.joblib")
    all_metrics["LogisticRegression"] = lr_metrics

    # 2. Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    joblib.dump(rf_model, MODEL_DIR / "baseline_rf.joblib")
    all_metrics["RandomForest"] = rf_metrics

    # 3. Naive Bayes
    nb_model, nb_metrics = train_naive_bayes(X_train, y_train, X_val, y_val)
    joblib.dump(nb_model, MODEL_DIR / "baseline_nb.joblib")
    all_metrics["MultinomialNB"] = nb_metrics

    # 4. LSTM (optional, memory-intensive)
    if not args.skip_lstm:
        lstm_model, lstm_metrics = train_lstm(X_train, y_train, X_val, y_val, num_classes, device=args.device)
        torch.save(lstm_model.state_dict(), MODEL_DIR / "lstm.pth")

