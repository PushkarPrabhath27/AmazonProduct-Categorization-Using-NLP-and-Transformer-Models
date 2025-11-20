"""Fine-tune BERT for product categorization.

Uses Hugging Face Transformers Trainer API with early stopping,
class weighting, and TensorBoard logging.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
BERT_MODEL_DIR = MODEL_DIR / "bert_final"
LOG_DIR = PROJECT_ROOT / "experiments" / "logs"
TENSORBOARD_DIR = LOG_DIR / "bert"
LOG_FILE = LOG_DIR / "train_bert.log"

BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """Load train and validation splits."""
    logging.info("Loading data from %s", PROCESSED_DIR)
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")

    # Encode labels
    le = LabelEncoder()
    train_df["label"] = le.fit_transform(train_df["category"])
    val_df["label"] = le.transform(val_df["category"])

    logging.info("Train: %d samples, Val: %d samples, Classes: %d", len(train_df), len(val_df), len(le.classes_))
    return train_df, val_df, le


def create_dataset(df: pd.DataFrame, tokenizer, max_length: int = 256) -> Dataset:
    """Create Hugging Face Dataset from dataframe."""
    texts = df["text_concat"].fillna("").tolist()
    labels = df["label"].tolist()

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    dataset_dict = {"text": texts, "labels": labels}
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return dataset


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_micro": f1_score(labels, predictions, average="micro"),
    }


class CustomTrainer(Trainer):
    """Custom trainer with class weighting support."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def train_bert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    le: LabelEncoder,
    *,
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    num_epochs: int = 5,
    weight_decay: float = 0.01,
    warmup_steps: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    use_class_weights: bool = True,
    device: str = "cpu",
) -> Dict[str, float]:
    """Fine-tune BERT model."""
    logging.info("Initializing BERT model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(le.classes_), problem_type="single_label_classification"
    )

    device_str = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device_str)

    # Create datasets
    logging.info("Creating datasets")
    train_dataset = create_dataset(train_df, tokenizer, max_length=max_length)
    val_dataset = create_dataset(val_df, tokenizer, max_length=max_length)

    # Compute class weights if requested
    class_weights = None
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight

        class_weights_array = compute_class_weight("balanced", classes=np.unique(train_df["label"]), y=train_df["label"])
        class_weights = torch.FloatTensor(class_weights_array)
        logging.info("Computed class weights: %s", class_weights)

    # Calculate warmup steps if not provided
    if warmup_steps is None:
        total_steps = (len(train_dataset) // (batch_size * gradient_accumulation_steps)) * num_epochs
        warmup_steps = int(0.1 * total_steps)
        logging.info("Calculated warmup_steps: %d (10%% of %d total steps)", warmup_steps, total_steps)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(BERT_MODEL_DIR / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_dir=str(TENSORBOARD_DIR),
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=1,
        report_to="tensorboard",
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )

    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[],
    )

    # Train
    logging.info("Starting training...")
    train_result = trainer.train()

    # Evaluate
    logging.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logging.info("Validation results: %s", eval_results)

    # Save final model and tokenizer
    logging.info("Saving model to %s", BERT_MODEL_DIR)
    trainer.save_model(str(BERT_MODEL_DIR))
    tokenizer.save_pretrained(str(BERT_MODEL_DIR))

    # Save label encoder
    import joblib

    joblib.dump(le, BERT_MODEL_DIR / "label_encoder.joblib")

    # Save training config
    config = {
        "model_name": model_name,
        "max_length": max_length,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "num_classes": len(le.classes_),
        "class_names": le.classes_.tolist(),
    }
    with open(BERT_MODEL_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return {
        "train_loss": train_result.training_loss,
        "val_f1_macro": eval_results.get("eval_f1_macro", 0.0),
        "val_f1_micro": eval_results.get("eval_f1_micro", 0.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for product categorization")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Pre-trained model name")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps (auto if None)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weighting")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples for training (debugging)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    train_df, val_df, le = load_data()

    if args.max_samples:
        logging.warning("SUBSAMPLING DATA TO %d SAMPLES (FAST MODE)", args.max_samples)
        train_df = train_df.head(args.max_samples)
        val_df = val_df.head(max(100, int(args.max_samples * 0.2)))


    metrics = train_bert(
        train_df,
        val_df,
        le,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_class_weights=not args.no_class_weights,
        device=args.device,
    )

    logging.info("Training completed. Final metrics: %s", metrics)
    logging.info("Model saved to %s", BERT_MODEL_DIR)
    logging.info("TensorBoard logs available at %s", TENSORBOARD_DIR)


if __name__ == "__main__":
    main()

