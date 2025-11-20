

You are an autonomous ML/NLP engineer. Work **sequentially** and produce artifacts at each checkpoint. Save code, logs, models, and the final report in the repository structure defined below. When a step finishes, save the described outputs and move to the next step. DO NOT skip any step. If anything in the dataset is ambiguous, make a conservative, documented choice and proceed; record that choice in `NOTES.md`.

---

## Repository structure (create these folders & files)

```
project/
├─ data/
│  ├─ raw/
│  │  └─ amazon_products.csv            # original downloaded file
│  └─ processed/
│     └─ train.csv, val.csv, test.csv
├─ notebooks/
│  ├─ 01-data-exploration.ipynb
│  ├─ 02-preprocessing.ipynb
│  ├─ 03-baseline-models.ipynb
│  ├─ 04-bert-finetune.ipynb
│  └─ 05-eval-and-interpretation.ipynb
├─ src/
│  ├─ data_loader.py
│  ├─ preprocess.py
│  ├─ feature_engineering.py
│  ├─ train_baselines.py
│  ├─ train_bert.py
│  ├─ eval.py
│  ├─ inference.py
│  └─ utils.py
├─ experiments/
│  └─ logs/ (store training logs, tensorboard)
├─ models/
│  ├─ baseline.joblib
│  └─ bert_final/
├─ results/
│  ├─ metrics.csv
│  ├─ confusion_matrix.png
│  └─ ROC_*.png
├─ REPORT/
│  └─ final_report.pdf (and final_report.md)
├─ environment.yml / requirements.txt
└─ README.md
```

---

## Environment & dependencies

* Use Python 3.10+.
* Required libraries (pin versions in `requirements.txt`):
  `pandas, numpy, scikit-learn, nltk, spacy, transformers, datasets, torch, tqdm, seaborn, matplotlib, optuna, joblib, tensorboard`
* Create `environment.yml` or `requirements.txt`. Provide exact pip install commands in README.

---

## Step 1 — Problem understanding & setup

1. Read the assignment document and summarise the problem in `README.md` (one paragraph) referencing that the dataset is the public Kaggle Amazon Products Dataset. Include target variable names: `product_title`, `product_description`, `category`. Save a copy of the assignment document in `data/raw/` or record its path.
2. Create a reproducible conda/venv environment with the `requirements.txt`. Save the environment creation command.

**Outputs:** `README.md`, `requirements.txt`, `environment.yml`, a `NOTES.md` that lists initial assumptions (e.g., how multi-word categories are handled).

---

## Step 2 — Data acquisition & quick EDA

1. Load the dataset from `data/raw/amazon_products.csv`. If CSV has multiple files, merge them into a single dataframe with the three columns (`product_title`, `product_description`, `category`) — drop or rename other columns but document the choice.
2. Print head, shape, column datatypes, number of unique categories, category counts, missing value counts.
3. Visualize:

   * Top 20 categories by frequency (bar plot).
   * Histogram of product_title length (tokens and chars).
   * Histogram of description length.
   * Class imbalance distribution (log-scale if necessary).
4. Save EDA notebook `notebooks/01-data-exploration.ipynb` and export PNGs to `results/`.

**Outputs:** `notebooks/01-data-exploration.ipynb`, `results/plots/*`, `data/raw/amazon_products.csv` (or pointer).

---

## Step 3 — Data cleaning & preprocessing

Follow the Methodology in the assignment precisely.

1. Text cleaning steps (implement in `src/preprocess.py` as reusable functions):

   * Remove HTML tags.
   * Normalize unicode (NFKC).
   * Lowercasing.
   * Remove punctuation and numbers (document option to keep numbers — implement flag `remove_numbers=True`).
   * Replace multiple whitespaces with single space.
   * Strip leading/trailing whitespace.
2. Tokenization:

   * Implement two tokenization routes:

     * Classic: `nltk.word_tokenize` or spaCy tokenizer (used for TF-IDF and LSTM).
     * Transformer route: use Hugging Face tokenizer (`bert-base-uncased`).
3. Stopword removal and lemmatization (optional controlled by flags); implement with `nltk` or `spaCy`.
4. Save a processed dataset and splits:

   * Shuffle dataset deterministically (set seed `42`).
   * Create train / val / test splits: 80% / 10% / 10% stratified by `category`.
   * Save CSVs to `data/processed/train.csv`, `val.csv`, `test.csv`.

**Outputs:** `src/preprocess.py`, `notebooks/02-preprocessing.ipynb`, processed CSVs.

---

## Step 4 — Feature representation (two parallel pipelines)

Implement code in `src/feature_engineering.py`.

A. **Traditional features:**

* TF-IDF vectorizer on `product_title + product_description` concatenated (use `max_features=50_000` or tune).
* N-grams: (1,2) and (1,3) experiments.
* Save fitted vectorizer to `models/tfidf_vectorizer.joblib`.

B. **Transformer embeddings:**

* Two options:

  1. Use `[CLS]` token pooled output from `bert-base-uncased` (or `sentence-transformers` if available) to create embeddings for each example. Save them as `.npy` for smaller experiments.
  2. Fine-tune BERT (see next step).
* If using BERT embeddings without fine-tuning, freeze the encoder and train a simple classifier (logistic regression) on embeddings as baseline.

**Outputs:** vectorizer file, embeddings saved in `data/processed/embeddings/`.

---

## Step 5 — Baseline model training & evaluation

Implement `src/train_baselines.py`.

1. **Baseline 1 — Logistic Regression (TF-IDF)**

   * Tune `C` with `GridSearchCV` (e.g., `[0.01, 0.1, 1, 10]`) and `max_iter=1000`.
   * Use stratified 5-fold CV on training data.
2. **Baseline 2 — Random Forest (TF-IDF)**

   * Tune `n_estimators=[100,300]`, `max_depth=[None, 20, 50]`.
3. **Baseline 3 — Naive Bayes (MultinomialNB)**
4. **Baseline 4 — LSTM (embedding + simple LSTM)**

   * Tokenize, pad sequences, embedding size 100 (random init), 1 LSTM layer (128 units), dropouts, early stopping.
   * Use class weighting or oversampling if heavy imbalance exists.

**Evaluation:**

* Use validation set for hyperparameter selection.
* Report for each model: accuracy, macro-precision, macro-recall, macro-F1, and per-class precision/recall/F1.
* Save best baseline model to `models/baseline.joblib` (or `models/lstm.pth` for LSTM).

**Outputs:** `notebooks/03-baseline-models.ipynb`, `models/*`, `results/metrics_baselines.csv`.

---

## Step 6 — Fine-tune BERT (State-of-the-art)

Implement `src/train_bert.py` using Hugging Face Transformers and `Trainer` or custom PyTorch loop.

**Key details:**

1. Model: `bert-base-uncased` (or `distilbert-base-uncased` for speed). For best results use BERT.
2. Inputs: concatenate `product_title` and `product_description` into a single text field separated by `[SEP]` (or use tokenizer pair inputs: title as `text_a`, description as `text_b`).
3. Hyperparameters (start with these and use Optuna to tune):

   * epochs: 3–5
   * learning_rate: 2e-5, 3e-5, 5e-5
   * batch_size: 16 or 32 (depending on GPU)
   * weight_decay: 0.01
   * warmup_steps: 500 or 0.1 * total_steps
   * gradient_accumulation_steps: if batch memory limited
4. Training strategies:

   * Use early stopping (patience 2 on validation F1).
   * Use class weighting or focal loss if heavy class imbalance.
   * Save checkpoints and the best model in `models/bert_final/`.
5. Logging:

   * Log training and validation loss and F1 per epoch.
   * Use TensorBoard (store in `experiments/logs/bert/`).

**Outputs:** `models/bert_final/` (model files), `experiments/logs/`, `notebooks/04-bert-finetune.ipynb`.

---

## Step 7 — Hyperparameter optimization

* Use `optuna` for either baseline (e.g., Random Forest params) or BERT (learning rate, batch size, weight decay).
* Limit trials to a reasonable number (e.g., 20 trials) with `direction='maximize'` applied to macro-F1 on validation set.
* Save best trial results to `experiments/optuna_results.json`.

**Outputs:** `experiments/optuna_results.json`, plots of optimization trends.

---

## Step 8 — Model evaluation on test set (final comparisons)

Implement `src/eval.py`.

1. Load the best models (best baseline and best BERT).
2. Run predictions on `data/processed/test.csv`.
3. Compute and save:

   * Confusion matrix (normalized & raw) `results/confusion_matrix.png`.
   * Classification report (`results/classification_report.txt` and `results/metrics.csv`) with per-class precision, recall, f1, support.
   * Macro and micro averaged metrics.
   * If feasible, compute top-2 and top-3 accuracy (because product categorization often benefits from top-k).
4. Plot ROC curves for multi-class using one-vs-rest (save `results/ROC_*.png`).
5. Compare models in a single summary table and highlight whether final accuracy meets the assignment target (≥ 85% for top categories). If target not reached, record analysis of why.

**Outputs:** `results/*` as above, `notebooks/05-eval-and-interpretation.ipynb`.

---

## Step 9 — Model interpretability & error analysis

1. Create confusion matrix analysis: list of top confused category pairs.
2. Per-category support vs F1 scatter plot (low support but high F1 etc.).
3. Use **LIME** or **SHAP** for BERT predictions on a small sample (20 hard examples).
4. Provide representative examples:

   * 10 correct predictions (with probabilities).
   * 10 incorrect predictions (show text, true label, predicted label, top-3 predicted classes).
5. Document whether title or description was more informative (do ablation: title-only, description-only, title+description and compare performance).

**Outputs:** `results/interpretability/*`, narrative in `REPORT/analysis.md`.

---

## Step 10 — Model saving & deployment artifacts

1. Save models:

   * Baseline: `models/baseline.joblib`.
   * BERT: full model + tokenizer in `models/bert_final/`.
2. Provide an inference script `src/inference.py` that:

   * Loads the model + vectorizer/tokenizer.
   * Exposes a `predict(text_title, text_description, top_k=3)` function returning top_k categories + probabilities.
   * Example CLI usage: `python src/inference.py --title "Apple iPhone 12" --desc "64GB, black" --top_k 3`
3. Provide a minimal Flask or FastAPI app skeleton (in `src/api/`) for real-time prediction (optional but recommended).
4. Document serialization formats and versioning.

**Outputs:** `src/inference.py`, `src/api/`, `models/*`.

---

## Step 11 — Documentation & final report

Create `REPORT/final_report.md` and export `final_report.pdf` with the following sections:

1. Executive summary (1 paragraph).
2. Dataset description and EDA (include key plots).
3. Preprocessing pipeline (exact steps & rationale).
4. Feature engineering approaches compared (TF-IDF vs BERT embeddings).
5. Model list & hyperparameters (tables).
6. Results: metrics tables, confusion matrices, ROC, top-k accuracy.
7. Interpretability & error analysis (examples).
8. Limitations and challenges (class imbalance, ambiguous categories, missing descriptions).
9. Next steps and deployment recommendations.
10. Appendix: code run commands, environment, and where artifacts are saved.

Also include a short notebook `notebooks/summary.ipynb` that reproduces final results (load saved model + test set, compute final metrics).

**Outputs:** `REPORT/final_report.md`, `REPORT/final_report.pdf`.

---

## Step 12 — Reproducibility checklist & delivery

* Add `run_all.sh` or `Makefile` that sequentially executes:

  1. `python src/data_loader.py` (download/verify dataset)
  2. `python src/preprocess.py` (create processed data)
  3. `python src/feature_engineering.py` (create vectorizers/embeddings)
  4. `python src/train_baselines.py` (train baselines)
  5. `python src/train_bert.py` (train BERT)
  6. `python src/eval.py` (evaluate)
  7. `python src/inference.py` (sample predictions)
* Provide sample `config.yaml` (or JSON) that contains: random seed, paths, model hyperparameters, and compute settings (useful for CI).
* Include `LICENSE` and `CONTRIBUTING.md` describing how to run experiments.

**Outputs:** `run_all.sh`/`Makefile`, `config.yaml`.

---

## Extra: performance targets & acceptance criteria

* Models must produce:

  * Baseline (TF-IDF + Logistic Regression) — report baseline metrics.
  * Fine-tuned BERT — expected to outperform baseline. Target: **≥ 85% accuracy** for top categories, or macro-F1 ≥ 0.80 (if dataset highly imbalanced). If target not reached, provide analysis and at least two remediation ideas.
* Provide top-k accuracy (top-3 recommended categories) — useful for e-commerce UX.

---

## Errors, logging & monitoring

* Every script must log progress (use `logging` module), save logs to `experiments/logs/<script>.log`.
* Catch I/O errors and provide helpful messages; e.g., if `amazon_products.csv` missing, log and exit with suggested download link.

---

## Sample code snippets (to include in `src/` files)

### Example: Concatenate title + description

```python
def combine_text(row):
    title = str(row['product_title']) if not pd.isna(row['product_title']) else ''
    desc  = str(row['product_description']) if not pd.isna(row['product_description']) else ''
    return title.strip() + " [SEP] " + desc.strip()
df['text'] = df.apply(combine_text, axis=1)
```

### Example: tokenization for BERT

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(list_texts, truncation=True, padding=True, max_length=256)
```

### Example: inference usage

```bash
python src/inference.py --title "Samsung Galaxy M32" --desc "6.4 inch, 64GB, 4GB RAM" --top_k 3
```

---

## Deliverables checklist (what you must hand in)

* Source code (all `src/` files & notebooks).
* Processed datasets (train/val/test CSVs).
* Saved models (`models/`).
* Results (plots, metrics CSVs).
* Final report PDF and markdown (`REPORT/final_report.pdf`, `final_report.md`).
* `README.md` with reproducible steps and environment.
* `NOTES.md` describing any deviations or assumptions.

---

## Final agent behavior instructions (how to run automatically)

* Work stepwise. After each major step, save artifacts and produce a short log message summarizing results (e.g., “BERT epoch 3 — val macro-F1 = 0.823 — checkpoint saved at models/bert_final/checkpoint-xxx”).
* If an experiment fails (OOM, convergence failure), attempt a safe fallback:

  * If BERT OOM: reduce `batch_size` by 2x or use `distilbert-base-uncased`.
  * If training does not converge: try lower learning rate and/or gradient clipping.
  * Document fallback choice in `NOTES.md`.
* At the end, generate `REPORT/final_report.pdf` and a `summary.txt` that lists final metrics for the user’s quick review.

---
Amazon Product Categorization - Complete Implementation Plan
Overview
This plan implements a 12-step ML/NLP pipeline for multi-class product categorization using Amazon product titles and descriptions. The system will train baseline models (Logistic Regression, Random Forest, Naive Bayes, LSTM) and fine-tune BERT, achieving target accuracy ≥85% for top categories.

Task Breakdown
Phase 1: Setup & Data Exploration (Steps 1-2)
Task 1.1: Repository Structure & Environment Setup

Create complete directory structure (data/, notebooks/, src/, experiments/, models/, results/, REPORT/)
Create requirements.txt with pinned versions of all dependencies
Create environment.yml for conda
Create initial README.md with problem summary
Create NOTES.md for assumptions and decisions
Copy assignment document to data/raw/
Task 1.2: Data Loading & EDA

Create src/data_loader.py to load and merge datasets from KaggleDocuents/
Move/copy amazon_products.csv to data/raw/
Create notebooks/01-data-exploration.ipynb:
Load dataset, inspect shape, dtypes, missing values
Analyze category distribution (unique counts, frequency)
Generate visualizations: top 20 categories, title/description length histograms, class imbalance
Save plots to results/plots/
Phase 2: Preprocessing (Step 3)
Task 2.1: Preprocessing Implementation

Create src/preprocess.py with functions:
HTML tag removal
Unicode normalization (NFKC)
Lowercasing, punctuation/numbers removal (configurable)
Whitespace normalization
Two tokenization routes: NLTK/spaCy (classic) and BERT tokenizer
Stopword removal and lemmatization (optional flags)
Create notebooks/02-preprocessing.ipynb demonstrating preprocessing
Create train/val/test splits (80/10/10, stratified, seed=42)
Save processed CSVs to data/processed/
Phase 3: Feature Engineering (Step 4)
Task 3.1: Feature Engineering Implementation

Create src/feature_engineering.py:
TF-IDF vectorization (max_features=50_000, n-grams (1,2) and (1,3))
Save vectorizer to models/tfidf_vectorizer.joblib
BERT embeddings extraction (frozen bert-base-uncased, [CLS] token pooling)
Save embeddings to data/processed/embeddings/ as .npy files
Phase 4: Baseline Models (Step 5)
Task 4.1: Baseline Model Training

Create src/train_baselines.py:
Logistic Regression with TF-IDF (GridSearchCV for C parameter)
Random Forest with TF-IDF (tune n_estimators, max_depth)
Multinomial Naive Bayes
LSTM model (embedding layer, 1 LSTM layer, 128 units, dropout, early stopping)
Create notebooks/03-baseline-models.ipynb
Evaluate on validation set, save metrics to results/metrics_baselines.csv
Save best models to models/ (baseline.joblib, lstm.pth)
Phase 5: BERT Fine-tuning (Step 6)
Task 5.1: BERT Training Implementation

Create src/train_bert.py:
Load bert-base-uncased model and tokenizer
Concatenate title + description with [SEP] token
Implement training loop with early stopping (patience=2)
Hyperparameters: epochs 3-5, lr 2e-5 to 5e-5, batch_size 16/32
Class weighting for imbalance
TensorBoard logging to experiments/logs/bert/
Save checkpoints and best model to models/bert_final/
Create notebooks/04-bert-finetune.ipynb
Phase 6: Hyperparameter Optimization (Step 7)
Task 6.1: Optuna Optimization

Integrate Optuna into training scripts
Optimize BERT hyperparameters (learning rate, batch size, weight decay)
Limit to 20 trials, maximize macro-F1 on validation set
Save results to experiments/optuna_results.json
Generate optimization trend plots
Phase 7: Final Evaluation (Step 8)
Task 7.1: Comprehensive Model Evaluation

Create src/eval.py:
Load best baseline and BERT models
Evaluate on test set
Generate confusion matrices (normalized & raw)
Compute classification reports (per-class and macro/micro averages)
Calculate top-2 and top-3 accuracy
Generate ROC curves (one-vs-rest for multi-class)
Save all outputs to results/
Create notebooks/05-eval-and-interpretation.ipynb
Compare models in summary table, check if ≥85% accuracy target met
Phase 8: Interpretability (Step 9)
Task 8.1: Error Analysis & Interpretability

Analyze confusion matrix for top confused category pairs
Create per-category support vs F1 scatter plot
Implement LIME/SHAP for BERT on 20 hard examples
Generate examples: 10 correct + 10 incorrect predictions with probabilities
Ablation study: title-only, description-only, title+description
Save outputs to results/interpretability/
Create REPORT/analysis.md with narrative
Phase 9: Deployment Artifacts (Step 10)
Task 9.1: Inference & API

Create src/inference.py:
Load models and tokenizers/vectorizers
Implement predict(text_title, text_description, top_k=3) function
CLI interface with argparse
Create src/api/ with Flask/FastAPI skeleton for real-time predictions
Document serialization formats in README
Phase 10: Documentation (Step 11)
Task 10.1: Final Report Generation

Create REPORT/final_report.md with all required sections:
Executive summary, dataset description, preprocessing, feature engineering
Model details & hyperparameters, results tables, interpretability
Limitations, next steps, appendix
Create notebooks/summary.ipynb to reproduce final results
Export REPORT/final_report.pdf (using markdown-to-pdf tool)
Phase 11: Reproducibility (Step 12)
Task 11.1: Automation & Configuration

Create run_all.sh or Makefile for sequential execution
Create config.yaml with seeds, paths, hyperparameters, compute settings
Create LICENSE file
Create CONTRIBUTING.md with experiment run instructions
Create src/utils.py for shared utilities and logging setup
Key Files to Create/Modify
Directory Structure:

data/raw/, data/processed/, data/processed/embeddings/
notebooks/ (5 notebooks + summary)
src/ (8 Python modules)
experiments/logs/
models/, results/, REPORT/
Core Implementation Files:

src/data_loader.py, src/preprocess.py, src/feature_engineering.py
src/train_baselines.py, src/train_bert.py, src/eval.py, src/inference.py, src/utils.py
src/api/ (Flask/FastAPI skeleton)
Configuration & Documentation:

requirements.txt, environment.yml, config.yaml
README.md, NOTES.md, LICENSE, CONTRIBUTING.md
run_all.sh or Makefile
REPORT/final_report.md, REPORT/final_report.pdf, REPORT/analysis.md
Success Criteria
All 12 steps completed with required outputs
BERT model achieves ≥85% accuracy (or macro-F1 ≥0.80) for top categories
All notebooks executable and reproducible
Complete documentation and deployment artifacts ready