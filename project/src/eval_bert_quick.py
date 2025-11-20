"""Quick BERT evaluation on test set."""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "bert_final"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

print("Loading BERT model and test data...")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Load test data
test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
le = joblib.load(MODEL_DIR / "label_encoder.joblib")

# Encode labels
y_test = le.transform(test_df["category"].values)

# Create text field (title + description)
test_df["text"] = test_df["product_title"].fillna("") + " " + test_df["product_description"].fillna("")

print(f"Test set: {len(test_df)} samples")

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

test_dataset = Dataset.from_pandas(test_df[["text"]])
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# Get predictions
print("Running predictions...")
y_pred = []
y_prob = []

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

with torch.no_grad():
    for i in range(0, len(test_dataset), 32):
        batch = test_dataset[i:i+32]
        inputs = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        y_pred.extend(preds)
        y_prob.extend(probs)

y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# Calculate metrics
print("\n" + "="*70)
print("BERT MODEL - TEST SET RESULTS")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
macro_precision = precision_score(y_test, y_pred, average="macro")
macro_recall = recall_score(y_test, y_pred, average="macro")
macro_f1 = f1_score(y_test, y_pred, average="macro")
micro_f1 = f1_score(y_test, y_pred, average="micro")

print(f"Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f} ({macro_f1*100:.2f}%)")
print(f"Micro F1:        {micro_f1:.4f} ({micro_f1*100:.2f}%)")
print("="*70)

# Save metrics
bert_metrics = {
    "model": "DistilBERT",
    "accuracy": accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "micro_f1": micro_f1
}

metrics_df = pd.DataFrame([bert_metrics])
metrics_df.to_csv(RESULTS_DIR / "metrics_bert_test.csv", index=False)

print(f"\nMetrics saved to: results/metrics_bert_test.csv")

# Classification report
print("\nPer-Category Performance:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save classification report
report = classification_report(y_test, y_pred, target_names=le.classes_)
with open(RESULTS_DIR / "classification_report_bert.txt", 'w') as f:
    f.write(report)

print(f"Classification report saved to: results/classification_report_bert.txt")
print("\nEvaluation complete!")
