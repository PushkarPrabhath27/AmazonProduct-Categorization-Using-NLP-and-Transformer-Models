"""
Utility to (re)generate notebooks/01-data-exploration.ipynb with the required
EDA steps described in Prompt.md. Run once whenever the notebook needs to be
rebuilt programmatically.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    notebook_path = Path("notebooks/01-data-exploration.ipynb")
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            "# 01 — Data Exploration\n\n"
            "Exploratory analysis of the Kaggle Amazon Products dataset after "
            "canonicalization (`data/raw/amazon_products_clean.csv`). "
            "This notebook validates required columns (`product_title`, "
            "`product_description`, `category`), summarises their distributions, "
            "and exports mandated plots plus summary statistics to "
            "`results/plots/` and `results/eda_summary.txt`."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "RAW_PATH = Path('data/raw/amazon_products_clean.csv')\n"
            "RESULTS_DIR = Path('results/plots')\n"
            "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
            "sns.set(style='whitegrid', font_scale=0.9)\n"
            "RAW_PATH"
        ),
        nbf.v4.new_code_cell(
            "df = pd.read_csv(RAW_PATH)\n"
            "print(f\"Shape: {df.shape}\")\n"
            "df.head()"
        ),
        nbf.v4.new_code_cell(
            "summary_lines = []\n"
            "summary_lines.append('Head:\\n' + df.head().to_string())\n"
            "summary_lines.append(f\"\\nShape: {df.shape}\")\n"
            "summary_lines.append('\\nDtypes:\\n' + df.dtypes.to_string())\n"
            "summary_lines.append('\\nMissing values:\\n' + df.isna().sum().to_string())\n"
            "summary_lines.append(f\"\\nUnique categories: {df['category'].nunique()}\")\n"
            "summary_lines.append('\\nTop 20 categories:\\n' + df['category'].value_counts().head(20).to_string())\n"
            "summary_text = '\\n'.join(summary_lines)\n"
            "Path('results/eda_summary.txt').write_text(summary_text, encoding='utf-8')\n"
            "summary_text.split('\\n')[:12]"
        ),
        nbf.v4.new_code_cell(
            "top20 = df['category'].value_counts().head(20)\n"
            "plt.figure(figsize=(12, 6))\n"
            "sns.barplot(x=top20.values, y=top20.index, palette='viridis')\n"
            "plt.title('Top 20 Categories by Frequency')\n"
            "plt.xlabel('Count')\n"
            "plt.ylabel('Category')\n"
            "plt.tight_layout()\n"
            "top20_path = RESULTS_DIR / 'top20_categories.png'\n"
            "plt.savefig(top20_path, dpi=200)\n"
            "plt.show()\n"
            "top20_path"
        ),
        nbf.v4.new_code_cell(
            "def length_stats(text_series):\n"
            "    char_len = text_series.fillna('').str.len()\n"
            "    token_len = text_series.fillna('').str.split().apply(len)\n"
            "    return char_len, token_len\n\n"
            "title_chars, title_tokens = length_stats(df['product_title'])\n"
            "desc_chars, desc_tokens = length_stats(df['product_description'])\n\n"
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
            "sns.histplot(title_tokens, bins=50, color='steelblue', ax=axes[0])\n"
            "axes[0].set_title('Product Title Length (tokens)')\n"
            "sns.histplot(title_chars, bins=50, color='coral', ax=axes[1])\n"
            "axes[1].set_title('Product Title Length (characters)')\n"
            "sns.histplot(desc_tokens, bins=50, color='darkgreen', ax=axes[2])\n"
            "axes[2].set_title('Product Description Length (tokens)')\n"
            "for ax in axes:\n"
            "    ax.set_xlabel('Count')\n"
            "fig.tight_layout()\n"
            "fig.savefig(RESULTS_DIR / 'title_description_length_hist.png', dpi=200)\n"
            "plt.show()\n\n"
            "cat_counts = df['category'].value_counts()\n"
            "plt.figure(figsize=(10, 5))\n"
            "sns.barplot(x=np.arange(len(cat_counts)), y=cat_counts.values, color='slateblue')\n"
            "plt.yscale('log')\n"
            "plt.title('Category Distribution (log scale)')\n"
            "plt.xlabel('Category index (sorted by freq)')\n"
            "plt.ylabel('Count (log scale)')\n"
            "imbalance_path = RESULTS_DIR / 'category_distribution_log.png'\n"
            "plt.tight_layout()\n"
            "plt.savefig(imbalance_path, dpi=200)\n"
            "plt.show()\n"
            "imbalance_path"
        ),
    ]

    nb["cells"] = cells
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, notebook_path)
    print(f"Wrote notebook to {notebook_path}")


if __name__ == "__main__":
    main()



