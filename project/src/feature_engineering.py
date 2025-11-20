"""Feature engineering routines for TF-IDF and transformer embeddings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBED_DIR = PROCESSED_DIR / "embeddings"
FEATURE_DIR = PROCESSED_DIR / "features"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "experiments" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "feature_engineering.log"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def load_split(name: str, processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected split '{name}' at {path}. Run preprocessing first.")
    logging.info("Loading %s split from %s", name, path)
    return pd.read_csv(path)


def run_tfidf(
    train_texts: Iterable[str],
    val_texts: Iterable[str],
    test_texts: Iterable[str],
    *,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 50_000,
) -> Dict[str, sparse.csr_matrix]:
    logging.info("Fitting TF-IDF vectorizer (ngram_range=%s, max_features=%d)", ngram_range, max_features)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    val_matrix = vectorizer.transform(val_texts)
    test_matrix = vectorizer.transform(test_texts)

    model_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, model_path)
    logging.info("Saved TF-IDF vectorizer to %s", model_path)

    sparse.save_npz(FEATURE_DIR / "tfidf_train.npz", train_matrix)
    sparse.save_npz(FEATURE_DIR / "tfidf_val.npz", val_matrix)
    sparse.save_npz(FEATURE_DIR / "tfidf_test.npz", test_matrix)
    logging.info("Serialized TF-IDF features to %s", FEATURE_DIR)

    return {"train": train_matrix, "val": val_matrix, "test": test_matrix}


def pooled_cls_embeddings(
    texts: Iterable[str],
    *,
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
    batch_size: int = 8,
    device: str | None = None,
    chunk_size: int = 1000,
    output_path: Path | None = None,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logging.info("Generating embeddings with %s on device %s (batch_size=%d, chunk_size=%d)", 
                 model_name, device, batch_size, chunk_size)
    texts = list(texts)
    total = len(texts)
    
    # If output_path provided, use memory-mapped array for incremental saving
    if output_path is not None:
        # Determine embedding dimension from a small sample
        sample_enc = tokenizer(
            texts[:1],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        sample_enc = {k: v.to(device) for k, v in sample_enc.items()}
        with torch.no_grad():
            sample_output = model(**sample_enc)
        embed_dim = sample_output.last_hidden_state.shape[-1]
        del sample_enc, sample_output
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Create memory-mapped array
        mmap_array = np.memmap(
            output_path,
            dtype=np.float32,
            mode="w+",
            shape=(total, embed_dim),
        )
        logging.info("Created memory-mapped array: shape=%s, dtype=%s", mmap_array.shape, mmap_array.dtype)
    else:
        mmap_array = None
        all_embeddings: list[np.ndarray] = []

    chunk_embeddings: list[np.ndarray] = []
    
    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            cls_states = outputs.last_hidden_state[:, 0, :]  # CLS token
            batch_emb = cls_states.cpu().numpy().astype(np.float32)
            
            # Clear GPU memory
            del enc, outputs, cls_states
            if device == "cuda":
                torch.cuda.empty_cache()
            
            if mmap_array is not None:
                # Write directly to memory-mapped array
                end_idx = min(start + batch_size, total)
                mmap_array[start:end_idx] = batch_emb
            else:
                chunk_embeddings.append(batch_emb)
            
            # Save chunk periodically to reduce memory
            if mmap_array is None and len(chunk_embeddings) * batch_size >= chunk_size:
                all_embeddings.extend(chunk_embeddings)
                chunk_embeddings.clear()
            
            if (start // batch_size) % 50 == 0:
                logging.info("Processed %d/%d examples", min(start + batch_size, total), total)
        
        # Handle remaining chunks
        if mmap_array is not None:
            mmap_array.flush()
            del mmap_array
            # Load as regular array
            result = np.memmap(output_path, dtype=np.float32, mode="r", shape=(total, embed_dim))
            result = np.array(result)  # Convert to regular array
        else:
            if chunk_embeddings:
                all_embeddings.extend(chunk_embeddings)
            result = np.vstack(all_embeddings)
    
    # Clean up
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return result


def run_bert_embeddings(
    splits: Dict[str, pd.DataFrame],
    *,
    text_column: str = "text_concat",
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
    batch_size: int = 8,
) -> None:
    for split_name, df in splits.items():
        logging.info("Creating BERT embeddings for %s (%d rows)", split_name, len(df))
        # Use temporary memory-mapped file for incremental saving
        temp_mmap_path = EMBED_DIR / f"{split_name}_{model_name.replace('/', '_')}.tmp.dat"
        final_path = EMBED_DIR / f"{split_name}_{model_name.replace('/', '_')}.npy"
        
        # Generate embeddings with memory-mapped output
        embeddings = pooled_cls_embeddings(
            df[text_column].fillna("").tolist(),
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            output_path=temp_mmap_path,
        )
        
        # Save embeddings (already converted from memmap to regular array)
        np.save(final_path, embeddings)
        if temp_mmap_path.exists():
            temp_mmap_path.unlink()  # Remove temporary file
        logging.info("Saved %s embeddings to %s (shape: %s)", split_name, final_path, embeddings.shape)
        del embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TF-IDF features and BERT embeddings.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR, help="Directory with processed CSV splits")
    parser.add_argument("--max-features", type=int, default=50_000, help="TF-IDF max features")
    parser.add_argument("--ngram-max", type=int, default=2, help="Upper bound for TF-IDF n-grams (lower bound fixed at 1)")
    parser.add_argument("--skip-tfidf", action="store_true", help="Skip TF-IDF feature generation")
    parser.add_argument("--skip-bert", action="store_true", help="Skip BERT embedding generation")
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased", help="Transformer model name for embeddings")
    parser.add_argument("--bert-max-length", type=int, default=256, help="Max sequence length for tokenizer")
    parser.add_argument("--bert-batch-size", type=int, default=8, help="Batch size for embedding extraction (reduced for memory efficiency)")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    splits = {name: load_split(name, args.processed_dir) for name in ("train", "val", "test")}

    if not args.skip_tfidf:
        run_tfidf(
            splits["train"]["text_concat"],
            splits["val"]["text_concat"],
            splits["test"]["text_concat"],
            ngram_range=(1, args.ngram_max),
            max_features=args.max_features,
        )
    else:
        logging.info("Skipping TF-IDF generation per flag.")

    if not args.skip_bert:
        run_bert_embeddings(
            splits,
            text_column="text_concat",
            model_name=args.bert_model,
            max_length=args.bert_max_length,
            batch_size=args.bert_batch_size,
        )
    else:
        logging.info("Skipping BERT embeddings per flag.")


if __name__ == "__main__":
    main()

