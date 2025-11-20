"""Utility to ingest Amazon product CSVs into canonical form.

Usage:
    python src/data_loader.py --products-path data/raw/amazon_products.csv \
        --output-path data/raw/amazon_products_clean.csv

The script verifies required columns (product_title, product_description, category),
adds missing ones when possible, and logs summary stats for downstream steps.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOG_DIR = PROJECT_ROOT / "experiments" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "data_loader.log"

REQUIRED_COLUMNS = ("product_title", "product_description", "category")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(stream=sys.stdout),
        ],
        force=True,
    )


def infer_column(df: pd.DataFrame, candidates: Iterable[str], fallback_name: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"None of the candidate columns {candidates} were found for '{fallback_name}'."
    )


def load_products(products_path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not products_path.exists():
        raise FileNotFoundError(
            f"Products file not found at {products_path}. Please download the Kaggle Amazon "
            "Products dataset and place the CSV in data/raw/."
        )

    logging.info("Loading products from %s", products_path)
    df = pd.read_csv(products_path, nrows=max_rows)
    logging.info("Loaded %d rows with columns: %s", len(df), list(df.columns))
    return df


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map: dict[str, str] = {}
    if "product_title" not in df.columns:
        column_map[infer_column(df, ["title", "name", "product_name"], "product_title")] = "product_title"
    if "product_description" not in df.columns:
        try:
            column_map[
                infer_column(
                    df,
                    ["description", "product_description", "desc", "details"],
                    "product_description",
                )
            ] = "product_description"
        except ValueError:
            logging.warning(
                "Product description not found; creating empty placeholder column so downstream steps can proceed."
            )
            df["product_description"] = ""
    if "category" not in df.columns:
        column_map[infer_column(df, ["category", "category_id", "category_name", "label"], "category")] = "category"

    if column_map:
        logging.info("Renaming columns: %s", column_map)
        df = df.rename(columns=column_map)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    # Keep only required columns for now; document drop in NOTES later if needed.
    df = df[list(REQUIRED_COLUMNS)].copy()
    return df



def merge_categories(df: pd.DataFrame, categories_path: Path | None) -> pd.DataFrame:
    if categories_path is None:
        return df

    if not categories_path.exists():
        raise FileNotFoundError(f"Category mapping file not found at {categories_path}.")

    cat_df = pd.read_csv(categories_path)
    logging.info("Loaded %d category rows", len(cat_df))

    key_col = infer_column(cat_df, ["id", "category_id"], "category_id")
    name_col = infer_column(cat_df, ["category", "category_name"], "category_name")

    if pd.api.types.is_numeric_dtype(df["category"]):
        logging.info("Merging numeric category IDs with names")
        df = df.merge(cat_df[[key_col, name_col]], left_on="category", right_on=key_col, how="left")
        df["category"] = df[name_col].fillna(df["category"].astype(str))
        df = df.drop(columns=[key_col, name_col])

    return df


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Saving cleaned dataset to %s", output_path)
    df.to_csv(output_path, index=False)
    logging.info("Head sample:\n%s", df.head())
    logging.info(
        "Dataset stats  rows: %d, unique categories: %d",
        len(df),
        df["category"].nunique(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amazon product data loader")
    parser.add_argument(
        "--products-path",
        type=Path,
        default=RAW_DIR / "amazon_products.csv",
        help="Path to the raw amazon_products.csv file",
    )
    parser.add_argument(
        "--categories-path",
        type=Path,
        default=None,
        help="Optional path to category metadata CSV with id/name columns",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=RAW_DIR / "amazon_products_clean.csv",
        help="Output CSV with canonical columns",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for development / memory-constrained runs",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    try:
        df = load_products(args.products_path, max_rows=args.max_rows)
        df = canonicalize_columns(df)
        df = merge_categories(df, args.categories_path)
        write_output(df, args.output_path)
    except Exception as exc:
        logging.exception("data_loader failed")
        raise


if __name__ == "__main__":
    main()
