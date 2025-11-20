"""Evaluate trained models on test set.

Generates comprehensive evaluation metrics, confusion matrices, ROC curves,
and top-k accuracy for model comparison.
"""

import os
import logging
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import joblib
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_test_data():
    """Load test set and features."""
    logging.info("Loading test data...")
    
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_test = sparse.load_npz(PROCESSED_DIR / "tfidf_test.npz")
    le = joblib.load(MODEL_DIR / "label_encoder.joblib")
    
    y_test = le.transform(test_df["category"].values)
    
    logging.info(f"Test set: {len(test_df)} samples, {len(le.classes_)} classes")
    
    return test_df, X_test, y_test, le


def evaluate_baseline(model_path, X_test, y_test, le, model_name="Baseline"):
    """Evaluate baseline model on test set."""
    logging.info(f"Evaluating {model_name}...")
    
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, average="macro"),
        "macro_recall": recall_score(y_test, y_pred, average="macro"),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "micro_f1": f1_score(y_test, y_pred, average="micro"),
    }
    
    # Top-k accuracy
    for k in [2, 3, 5]:
        if len(le.classes_) >= k:
            top_k_acc = top_k_accuracy_score(y_test, y_prob, k=k)
            metrics[f"top_{k}_accuracy"] = top_k_acc
    
    logging.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
    
    return metrics, y_pred, y_prob


def evaluate_bert(model_dir, test_df, le):
    """Evaluate BERT model on test set."""
    logging.info("Evaluating BERT model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Prepare dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
        test_dataset = Dataset.from_pandas(test_df)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        test_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
        
        # Get predictions
        y_pred = []
        y_prob = []
        
        with torch.no_grad():
            for i in range(0, len(test_dataset), 32):
                batch = test_dataset[i:i+32]
                inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
                
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                y_pred.extend(preds)
                y_prob.extend(probs)
        
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        y_test = le.transform(test_df["category"].values)
        
        metrics = {
            "model": "BERT",
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_precision": precision_score(y_test, y_pred, average="macro"),
            "macro_recall": recall_score(y_test, y_pred, average="macro"),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "micro_f1": f1_score(y_test, y_pred, average="micro"),
        }
        
        # Top-k accuracy
        for k in [2, 3, 5]:
            if len(le.classes_) >= k:
                top_k_acc = top_k_accuracy_score(y_test, y_prob, k=k)
                metrics[f"top_{k}_accuracy"] = top_k_acc
        
        logging.info(f"BERT - Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
        
        return metrics, y_pred, y_prob
        
    except Exception as e:
        logging.warning(f"Could not evaluate BERT model: {e}")
        return None, None, None


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", filename="confusion_matrix.png"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds', xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved confusion matrix to {filename}")


def plot_roc_curves(y_true, y_prob, labels, filename="ROC_curves.png"):
    """Plot ROC curves for multi-class classification."""
    n_classes = len(labels)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot top 10 classes by support
    plt.figure(figsize=(12, 8))
    
    # Get top classes by frequency
    from collections import Counter
    class_counts = Counter(y_true)
    top_classes = [cls for cls, _ in class_counts.most_common(10)]
    
    for i in top_classes:
        plt.plot(fpr[i], tpr[i], lw=2, 
                label=f'{labels[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Top 10 Categories)')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved ROC curves to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on test set")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate baseline only")
    parser.add_argument("--bert-only", action="store_true", help="Evaluate BERT only")
    args = parser.parse_args()
    
    # Load test data
    test_df, X_test, y_test, le = load_test_data()
    
    all_metrics = []
    predictions = {}
    probabilities = {}
    
    # Evaluate baseline
    if not args.bert_only:
        baseline_path = MODEL_DIR / "baseline.joblib"
        if baseline_path.exists():
            metrics, y_pred, y_prob = evaluate_baseline(
                baseline_path, X_test, y_test, le, "Best Baseline (LR)"
            )
            all_metrics.append(metrics)
            predictions["baseline"] = y_pred
            probabilities["baseline"] = y_prob
            
            # Generate confusion matrix
            plot_confusion_matrix(
                y_test, y_pred, le.classes_,
                title="Best Baseline Confusion Matrix",
                filename="confusion_matrix_baseline.png"
            )
            
            # Generate ROC curves
            plot_roc_curves(y_test, y_prob, le.classes_, filename="ROC_baseline.png")
    
    # Evaluate BERT
    if not args.baseline_only:
        bert_dir = MODEL_DIR / "bert_final"
        if bert_dir.exists():
            metrics, y_pred, y_prob = evaluate_bert(bert_dir, test_df, le)
            if metrics:
                all_metrics.append(metrics)
                predictions["bert"] = y_pred
                probabilities["bert"] = y_prob
                
                # Generate confusion matrix
                plot_confusion_matrix(
                    y_test, y_pred, le.classes_,
                    title="BERT Confusion Matrix",
                    filename="confusion_matrix_bert.png"
                )
                
                # Generate ROC curves
                plot_roc_curves(y_test, y_prob, le.classes_, filename="ROC_bert.png")
    
    # Save metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(RESULTS_DIR / "metrics_test.csv", index=False)
        logging.info(f"Saved test metrics to metrics_test.csv")
        
        print("\n" + "="*70)
        print("TEST SET EVALUATION RESULTS")
        print("="*70)
        print(metrics_df.to_string(index=False))
        print("="*70)
        
        # Check if target met
        for _, row in metrics_df.iterrows():
            if row['accuracy'] >= 0.85:
                print(f"\n✅ {row['model']}: Target accuracy ≥85% ACHIEVED ({row['accuracy']:.2%})")
            else:
                print(f"\n⚠️  {row['model']}: Below target ({row['accuracy']:.2%})")
    
    # Generate classification reports
    for model_name, y_pred in predictions.items():
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        report_path = RESULTS_DIR / f"classification_report_{model_name}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logging.info(f"Saved classification report to {report_path}")
    
    logging.info("Evaluation complete!")


if __name__ == "__main__":
    main()
