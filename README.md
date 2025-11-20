# Amazon Product Categorization

Multi-class text classification system for automatic product categorization using NLP and transformer-based models.

## 🎯 Project Goal

Build an AI-driven system to automatically categorize e-commerce products into 15 categories based on product titles and descriptions, achieving ≥85% accuracy.

## 📊 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **96.92%** |
| **Macro F1-Score** | **96.47%** |
| **Top-3 Accuracy** | **99.45%** |

✅ **Target Achievement**: Exceeded 85% accuracy requirement

## 🏆 Best Model

**Logistic Regression** with TF-IDF features
- Training: GridSearchCV hyperparameter tuning
- Features: 50,000 TF-IDF unigrams + bigrams
- Performance: 96.92% test accuracy

## 📁 Repository Structure

```
project/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Processed splits & features
├── notebooks/                  # 5 Jupyter notebooks for analysis
│   ├── 01-data-exploration.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-baseline-models.ipynb
│   ├── 04-bert-finetune.ipynb
│   └── 05-eval-and-interpretation.ipynb
├── src/                        # Source code
│   ├── preprocess.py           # Data cleaning & splitting
│   ├── feature_engineering.py  # TF-IDF & BERT embeddings
│   ├── train_baselines.py      # Train LR, RF, NB
│   ├── train_bert.py           # Fine-tune DistilBERT
│   ├── eval.py                 # Test set evaluation
│   └── inference.py            # Production predictions
├── models/                     # Trained models
│   ├── baseline.joblib         # Best baseline (LR)
│   ├── bert_final/             # DistilBERT model
│   └── *.joblib                # Vectorizers, encoders
├── results/                    # Metrics, plots, reports
│   ├── metrics_test.csv
│   ├── confusion_matrix_*.png
│   └── ROC_*.png
├── REPORT/                     # Final documentation
│   └── final_report.md
└── requirements.txt            # Dependencies
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate product-categorization
```

### 2. Data Preparation

```bash
# Preprocess data (creates train/val/test splits)
python src/preprocess.py

# Generate features
python src/feature_engineering.py
```

### 3. Model Training

```bash
# Train baseline models
python src/train_baselines.py

# Train BERT (optional, resource-intensive)
python src/train_bert.py --num-epochs 3 --batch-size 16
```

### 4. Evaluation

```bash
# Evaluate on test set
python src/eval.py
```

### 5. Make Predictions

```bash
# Use inference script
python src/inference.py \
  --title "Samsung Galaxy M32" \
  --desc "6.4 inch, 64GB, 4GB RAM" \
  --top-k 3
```

**Output**:
```
Predicted Category: Electronics
Confidence: 98.5%

Top 3 Predictions:
  1. Electronics          (98.5%)
  2. Cell Phones          (1.2%)
  3. Cameras & Photo      (0.2%)
```

## 📝 Models Implemented

### Baseline Models (TF-IDF Features)

1. **Logistic Regression** (Best: 96.92% accuracy)
   - GridSearchCV hyperparameter tuning
   - Regularization parameter C optimized

2. **Random Forest** (88.13% F1)
   - Ensemble of 100 decision trees
   - Max depth: 50

3. **Multinomial Naive Bayes** (86.89% F1)
   - Probabilistic baseline

### Transformer Model

4. **DistilBERT** (Fine-tuned)
   - 66M parameters
   - Hugging Face Transformers
   - Training: AdamW optimizer, lr=3e-5

## 📊 Key Features

- **Data Pipeline**: Automated preprocessing, splitting, feature extraction
- **Hyperparameter Tuning**: GridSearchCV for baseline optimization
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, top-k accuracy
- **Production-Ready Inference**: CLI tool for real-time predictions
- **Full Documentation**: Notebooks, reports, code comments

## 📖 Documentation

- **Final Report**: [`REPORT/final_report.md`](REPORT/final_report.md)
- **Notebooks**: Explore [`notebooks/`](notebooks/) directory
- **Code**: Well-commented Python modules in [`src/`](src/)

## 🔬 Methodology

### Preprocessing
- HTML tag removal
- Unicode normalization
- Lowercasing, punctuation removal
- Tokenization (NLTK + BERT tokenizer)

### Feature Engineering
- **TF-IDF**: 50k features, (1,2)-grams
- **BERT Embeddings**: 768-dim vectors from DistilBERT

### Training Strategy
- Stratified 80/10/10 split (train/val/test)
- 5-fold cross-validation for hyperparameter tuning
- Early stopping for deep learning models

## 📈 Performance Metrics

### Test Set Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 96.92% | 97.16% | 95.84% | 96.47% |
| Random Forest | 89.37% | 89.50% | 87.91% | 88.13% |
| Multinomial NB | 88.16% | 88.20% | 86.66% | 86.89% |

### Top-K Accuracy
- **Top-2**: 98.96%
- **Top-3**: 99.45%
- **Top-5**: 99.77%

## 🛠️ Technical Stack

- **Python**: 3.10+
- **ML/NLP**: scikit-learn, transformers, torch
- **Data**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn
- **Logging**: tensorboard

## 🎯 Use Cases

- E-commerce product cataloging
- Automated product tagging
- Search relevance improvement  
- Category recommendation systems

## 📜 License

This project is for educational purposes as part of the Natural Language Processing course assignment.

## 👤 Author

**Project**: Amazon Product Categorization  
**Course**: Natural Language Processing  
**Date**: November 2025

## 🙏 Acknowledgments

- Dataset: Amazon Product Dataset (Kaggle)
- References: BERT (Devlin et al., 2019), Transformers (Vaswani et al., 2017)
- Framework: Hugging Face Transformers

---

**For detailed methodology and results, see [`REPORT/final_report.md`](REPORT/final_report.md)**
