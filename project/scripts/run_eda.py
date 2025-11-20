import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

print("Running EDA helper script...")

RAW = Path('data/raw/amazon_products_clean.csv')
if not RAW.exists():
    raise SystemExit('Clean dataset missing at data/raw/amazon_products_clean.csv')

df = pd.read_csv(RAW)

plots_dir = Path('results/plots')
plots_dir.mkdir(parents=True, exist_ok=True)
summary_path = Path('results/eda_summary.txt')

summary_lines = []
summary_lines.append("Head:\n" + df.head().to_string())
summary_lines.append(f"\nShape: {df.shape}")
summary_lines.append("\nDtypes:\n" + df.dtypes.to_string())
summary_lines.append("\nMissing values:\n" + df.isna().sum().to_string())
summary_lines.append(f"\nUnique categories: {df['category'].nunique()}")
summary_lines.append("\nTop 20 categories:\n" + df['category'].value_counts().head(20).to_string())

summary_path.write_text("\n".join(summary_lines), encoding='utf-8')

sns.set(style='whitegrid', font_scale=0.9)

plt.figure(figsize=(12, 6))
top20 = df['category'].value_counts().head(20)
sns.barplot(x=top20.values, y=top20.index, palette='viridis')
plt.title('Top 20 Categories by Frequency')
plt.xlabel('Count')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig(plots_dir / 'top20_categories.png', dpi=200)
plt.close()


def length_stats(text_series):
    char_len = text_series.fillna('').str.len()
    token_len = text_series.fillna('').str.split().apply(len)
    return char_len, token_len


title_chars, title_tokens = length_stats(df['product_title'])
desc_chars, desc_tokens = length_stats(df['product_description'])

plt.figure(figsize=(10, 5))
sns.histplot(title_tokens, bins=50, color='steelblue')
plt.title('Product Title Length (tokens)')
plt.xlabel('Tokens per title')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(plots_dir / 'title_length_tokens.png', dpi=200)
plt.close()

plt.figure(figsize=(10, 5))
sns.histplot(title_chars, bins=50, color='coral')
plt.title('Product Title Length (characters)')
plt.xlabel('Characters per title')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(plots_dir / 'title_length_characters.png', dpi=200)
plt.close()

plt.figure(figsize=(10, 5))
sns.histplot(desc_tokens, bins=50, color='darkgreen')
plt.title('Product Description Length (tokens)')
plt.xlabel('Tokens per description')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(plots_dir / 'description_length_tokens.png', dpi=200)
plt.close()

plt.figure(figsize=(10, 5))
cat_counts = df['category'].value_counts()
sns.barplot(x=np.arange(len(cat_counts)), y=cat_counts.values, color='slateblue')
plt.yscale('log')
plt.title('Category Distribution (log scale)')
plt.xlabel('Category index (sorted by freq)')
plt.ylabel('Count (log scale)')
plt.tight_layout()
plt.savefig(plots_dir / 'category_distribution_log.png', dpi=200)
plt.close()

print('EDA plots saved to', plots_dir.resolve())
print('EDA summary saved to', summary_path.resolve())
