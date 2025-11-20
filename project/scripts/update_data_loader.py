from pathlib import Path
import re

path = Path('src/data_loader.py')
text = path.read_text()
pattern = re.compile(r"def canonicalize_columns\(df: pd\.DataFrame\) -> pd\.DataFrame:\s+column_map = \{\}(.*?)return df", re.S)
match = pattern.search(text)
if not match:
    raise SystemExit('pattern not found')
replacement = """def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:\n    column_map: dict[str, str] = {}\n    if \"product_title\" not in df.columns:\n        column_map[infer_column(df, [\"title\", \"name\", \"product_name\"], \"product_title\")] = \"product_title\"\n    if \"product_description\" not in df.columns:\n        try:\n            column_map[\n                infer_column(\n                    df,\n                    [\"description\", \"product_description\", \"desc\", \"details\"],\n                    \"product_description\",\n                )\n            ] = \"product_description\"\n        except ValueError:\n            logging.warning(\n                \"Product description not found; creating empty placeholder column so downstream steps can proceed.\"\n            )\n            df[\"product_description\"] = \"\"\n    if \"category\" not in df.columns:\n        column_map[infer_column(df, [\"category\", \"category_name\", \"label\"], \"category\")] = \"category\"\n\n    if column_map:\n        logging.info(\"Renaming columns: %s\", column_map)\n        df = df.rename(columns=column_map)\n\n    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]\n    if missing:\n        raise ValueError(f\"Missing required columns after normalization: {missing}\")\n\n    # Keep only required columns for now; document drop in NOTES later if needed.\n    df = df[list(REQUIRED_COLUMNS)].copy()\n    return df\n"""
text = pattern.sub(replacement, text, count=1)
path.write_text(text)
