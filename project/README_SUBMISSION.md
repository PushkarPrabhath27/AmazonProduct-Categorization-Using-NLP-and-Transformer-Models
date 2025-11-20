# Amazon Product Categorization Project - Submission

## Project Overview
This project implements a multi-class product categorization system using Amazon product titles and descriptions. It includes a complete ML pipeline from data preprocessing to model training and evaluation.

## Completed Steps
1.  **Data Acquisition & Exploration**: Loaded and analyzed the Amazon products dataset.
2.  **Preprocessing**: Implemented text cleaning and train/val/test splitting.
3.  **Feature Engineering**: Generated TF-IDF features and BERT embeddings.
4.  **Baseline Models**: Trained and evaluated Logistic Regression, Random Forest, and Naive Bayes.
    -   **Best Baseline**: Logistic Regression (95.69% Macro-F1)
5.  **BERT Fine-tuning**: Implemented a fine-tuning pipeline for BERT/DistilBERT.

## "Fast Mode" Configuration
Due to hardware constraints and time limits, the BERT fine-tuning was executed in a **"Fast Mode"** configuration for demonstration purposes:
-   **Model**: `distilbert-base-uncased` (Lighter version of BERT)
-   **Training Data**: Subsampled to **100 samples** (Proof of concept)
-   **Epochs**: 1
-   **Batch Size**: 8

**Note**: The code is fully capable of training on the full dataset (80,000 samples) by removing the `--max-samples` flag.

## Key Results
### Baseline Models (Full Dataset)
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 96.49% | 95.69% |
| Random Forest | 89.37% | 88.13% |
| Naive Bayes | 88.16% | 86.89% |

### BERT Model
-   The BERT model included in `models/bert_final` is a **demo checkpoint** trained on 100 samples.
-   To train the full model, run:
    ```bash
    python src/train_bert.py --num-epochs 3 --batch-size 16
    ```

## Repository Structure
-   `src/`: Source code for preprocessing, training, and evaluation.
-   `notebooks/`: Jupyter notebooks for exploration and analysis.
-   `models/`: Saved model artifacts.
-   `results/`: Evaluation metrics (CSV).
-   `experiments/logs/`: Training logs.

## How to Run
1.  **Install Dependencies**: `pip install -r requirements.txt`
2.  **Run Baseline Training**: `python src/train_baselines.py`
3.  **Run BERT Training (Fast Mode)**:
    ```bash
    python src/train_bert.py --model-name distilbert-base-uncased --max-samples 100 --num-epochs 1 --no-class-weights
    ```
