'''Text preprocessing utilities and dataset splitting pipeline.

Usage:
    python src/preprocess.py --input data/raw/amazon_products_clean.csv --output-dir data/processed
'''

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
LOG_DIR = PROJECT_ROOT / 'experiments' / 'logs'
NLTK_DATA_DIR = PROJECT_ROOT / '.nltk_data'
for directory in (LOG_DIR, NLTK_DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)
nltk.data.path = [str(NLTK_DATA_DIR)]
LOG_FILE = LOG_DIR / 'preprocess.log'


def ensure_nltk_resource(package: str, resource_path: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package, quiet=True, download_dir=str(NLTK_DATA_DIR))


ensure_nltk_resource('punkt', 'tokenizers/punkt')
ensure_nltk_resource('stopwords', 'corpora/stopwords')
ensure_nltk_resource('wordnet', 'corpora/wordnet')
ensure_nltk_resource('omw-1.4', 'corpora/omw-1.4')

CLEAN_COLS = ('product_title', 'product_description')


@dataclass
class PreprocessConfig:
    remove_numbers: bool = True
    lowercase: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = False
    seed: int = 42


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()],
        force=True,
    )


def strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', ' ', text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def basic_clean(text: str, *, remove_numbers: bool, lowercase: bool) -> str:
    if not isinstance(text, str):
        text = '' if pd.isna(text) else str(text)
    text = strip_html(text)
    text = unicodedata.normalize('NFKC', text)
    if lowercase:
        text = text.lower()
    allowed = r'[^a-z\s]' if remove_numbers else r'[^a-z0-9\s]'
    text = re.sub(allowed, ' ', text)
    return normalize_whitespace(text)


def classic_tokenize(text: str) -> List[str]:
    from nltk import word_tokenize

    if not text:
        return []
    return word_tokenize(text)


def transformer_tokenize(texts: Sequence[str], model_name: str = 'bert-base-uncased', max_length: int = 256) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )


def apply_cleaning(df: pd.DataFrame, cfg: PreprocessConfig) -> pd.DataFrame:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stopword_set = set(stopwords.words('english')) if cfg.remove_stopwords else None
    lemmatizer = WordNetLemmatizer() if cfg.lemmatize else None

    def _post_process(text: str) -> str:
        tokens = text.split()
        if stopword_set is not None:
            tokens = [tok for tok in tokens if tok not in stopword_set]
        if lemmatizer is not None:
            tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
        return ' '.join(tokens)

    df = df.copy()
    for col in CLEAN_COLS:
        df[col] = df[col].fillna('')
        df[col] = df[col].apply(
            lambda x: basic_clean(x, remove_numbers=cfg.remove_numbers, lowercase=cfg.lowercase)
        )
        if cfg.remove_stopwords or cfg.lemmatize:
            df[col] = df[col].apply(_post_process)

    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    df['text_concat'] = (df['product_title'].str.strip() + ' [SEP] ' + df['product_description'].str.strip()).str.strip()
    return df


def make_splits(df: pd.DataFrame, *, stratify_col: str = 'category', seed: int = 42):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[stratify_col],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df[stratify_col],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_frames(frames: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        path = output_dir / f"{name}.csv"
        LOGGER.info('Saving %s with %d rows to %s', name, len(frame), path)
        frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Clean text fields and create stratified dataset splits.')
    parser.add_argument('--input', type=Path, default=RAW_DIR / 'amazon_products_clean.csv', help='Path to canonical raw CSV')
    parser.add_argument('--output-dir', type=Path, default=PROCESSED_DIR, help='Directory to store processed CSVs')
    parser.add_argument('--keep-numbers', action='store_true', help='Retain digits during cleaning.')
    parser.add_argument('--no-lowercase', action='store_true', help='Disable lowercasing.')
    parser.add_argument('--remove-stopwords', action='store_true', help='Remove English stopwords after basic cleaning.')
    parser.add_argument('--lemmatize', action='store_true', help='Apply WordNet lemmatization after cleaning (implies tokenization).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling and splits.')
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = PreprocessConfig(
        remove_numbers=not args.keep_numbers,
        lowercase=not args.no_lowercase,
        remove_stopwords=args.remove_stopwords,
        lemmatize=args.lemmatize,
        seed=args.seed,
    )

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found at {args.input}. Please run data_loader first.")

    LOGGER.info('Loading dataset from %s', args.input)
    df = pd.read_csv(args.input)
    LOGGER.info('Loaded %d rows', len(df))

    df = apply_cleaning(df, cfg)
    LOGGER.info('Finished cleaning text fields.')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed_path = args.output_dir / 'processed_full.csv'
    df.to_csv(processed_path, index=False)

    train_df, val_df, test_df = make_splits(df, stratify_col='category', seed=cfg.seed)
    LOGGER.info('Split sizes  train: %d, val: %d, test: %d', len(train_df), len(val_df), len(test_df))

    save_frames({'train': train_df, 'val': val_df, 'test': test_df}, args.output_dir)


if __name__ == '__main__':
    main()
