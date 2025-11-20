# Contributing to Amazon Product Categorization Project

## Running Experiments

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or using conda:
   ```bash
   conda env create -f environment.yml
   conda activate amazon-product-cat
   ```

2. Ensure the dataset is available at `data/raw/amazon_products.csv`

### Complete Pipeline

Run the entire pipeline using one of these methods:

**Windows (PowerShell):**
```powershell
.\run_all.ps1
```

**Linux/Mac (Bash):**
```bash
bash run_all.sh
```

**Using Make:**
```bash
make all
```

### Individual Steps

You can also run individual steps:

1. **Data Loading:**
   ```bash
   python src/data_loader.py
   ```

2. **Preprocessing:**
   ```bash
   python src/preprocess.py
   ```

3. **Feature Engineering:**
   ```bash
   python src/feature_engineering.py
   ```

4. **Train Baselines:**
   ```bash
   python src/train_baselines.py
   ```

5. **Fine-tune BERT:**
   ```bash
   python src/train_bert.py
   ```

6. **Evaluate:**
   ```bash
   python src/eval.py
   ```

7. **Inference:**
   ```bash
   python src/inference.py --title "Product Title" --desc "Description" --top-k 3
   ```

## Configuration

Modify `config.yaml` to adjust hyperparameters, paths, and compute settings.

## Logging

All scripts log to `experiments/logs/<script_name>.log`. Check these files for detailed execution information.

## Model Artifacts

- Baseline models: `models/baseline*.joblib`
- BERT model: `models/bert_final/`
- TF-IDF vectorizer: `models/tfidf_vectorizer.joblib`
- Label encoder: `models/label_encoder.joblib`

## Results

- Metrics: `results/metrics.csv`, `results/metrics_baselines.csv`
- Plots: `results/*.png`
- Reports: `results/classification_report_*.txt`

## Notebooks

Interactive notebooks are available in `notebooks/`:
- `01-data-exploration.ipynb` - EDA
- `02-preprocessing.ipynb` - Preprocessing steps
- `03-baseline-models.ipynb` - Baseline model training
- `04-bert-finetune.ipynb` - BERT fine-tuning
- `05-eval-and-interpretation.ipynb` - Evaluation and analysis

## Troubleshooting

### Memory Issues

If you encounter out-of-memory errors:

1. **BERT Embeddings:** Reduce `--bert-batch-size` in `feature_engineering.py`
2. **BERT Training:** Reduce `--batch-size` and increase `--gradient-accumulation-steps` in `train_bert.py`
3. **Baseline Training:** Use `--skip-lstm` flag to skip memory-intensive LSTM training

### Missing Dependencies

Ensure all packages in `requirements.txt` are installed. If using NLTK, ensure required data is downloaded:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to functions and classes
- Log important operations

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Error messages and stack traces
- Relevant log files from `experiments/logs/`

