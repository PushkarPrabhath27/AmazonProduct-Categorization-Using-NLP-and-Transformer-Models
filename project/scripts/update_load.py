from pathlib import Path
import textwrap

path = Path('src/data_loader.py')
text = path.read_text()
old = """def load_products(products_path: Path) -> pd.DataFrame:\n    if not products_path.exists():\n        raise FileNotFoundError(\n            f\"Products file not found at {products_path}. Please download the Kaggle Amazon \"\n            \"Products dataset and place the CSV in data/raw/.\"\n        )\n\n    logging.info(\"Loading products from %s\", products_path)\n    df = pd.read_csv(products_path)\n    logging.info(\"Loaded %d rows with columns: %s\", len(df), list(df.columns))\n    return df\n"""
new = """def load_products(products_path: Path, max_rows: int | None = None) -> pd.DataFrame:\n    if not products_path.exists():\n        raise FileNotFoundError(\n            f\"Products file not found at {products_path}. Please download the Kaggle Amazon \"\n            \"Products dataset and place the CSV in data/raw/.\"\n        )\n\n    logging.info(\"Loading products from %s\", products_path)\n    df = pd.read_csv(products_path, nrows=max_rows)\n    logging.info(\"Loaded %d rows with columns: %s\", len(df), list(df.columns))\n    return df\n"""
if old not in text:
    raise SystemExit('load block not found')
path.write_text(text.replace(old, new))
