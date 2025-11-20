from pathlib import Path
import pandas as pd

RAW = Path('data/raw/amazon_products_clean.csv')
if not RAW.exists():
    raise SystemExit('Missing cleaned dataset. Run data_loader first.')

summary_path = Path('results/eda_summary.txt')
df = pd.read_csv(RAW)
with summary_path.open('w', encoding='utf-8') as f:
    f.write(f"Shape: {df.shape}\n")
    f.write(f"dtypes:\n{df.dtypes}\n\n")
    f.write(f"Missing values:\n{df.isna().sum()}\n\n")
    f.write(f"Unique categories: {df['category'].nunique()}\n\n")
    f.write('Top 20 categories by frequency:\n')
    f.write(f"{df['category'].value_counts().head(20)}\n")
print('EDA summary saved to', summary_path)
