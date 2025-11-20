# Makefile for Amazon Product Categorization Pipeline

.PHONY: all data preprocess features baselines bert eval inference clean help

PYTHON := python
PYTHON_CMD := $(PYTHON)

help:
	@echo "Available targets:"
	@echo "  all        - Run complete pipeline"
	@echo "  data       - Load and clean data"
	@echo "  preprocess - Preprocess data and create splits"
	@echo "  features   - Generate TF-IDF and BERT embeddings"
	@echo "  baselines  - Train baseline models"
	@echo "  bert       - Fine-tune BERT"
	@echo "  eval       - Evaluate models on test set"
	@echo "  inference  - Run sample inference"
	@echo "  clean      - Clean generated files"
	@echo "  help       - Show this help message"

all: data preprocess features baselines bert eval inference

data:
	$(PYTHON_CMD) src/data_loader.py

preprocess:
	$(PYTHON_CMD) src/preprocess.py

features:
	$(PYTHON_CMD) src/feature_engineering.py

baselines:
	$(PYTHON_CMD) src/train_baselines.py --skip-lstm

bert:
	$(PYTHON_CMD) src/train_bert.py

eval:
	$(PYTHON_CMD) src/eval.py

inference:
	$(PYTHON_CMD) src/inference.py --title "Apple iPhone 12" --desc "64GB, black" --top-k 3 --model baseline

clean:
	@echo "Cleaning generated files..."
	rm -rf data/processed/*.csv data/processed/features/* data/processed/embeddings/*
	rm -rf models/*.joblib models/*.pth models/bert_final/*
	rm -rf results/*
	rm -rf experiments/logs/*.log
	@echo "Clean complete (preserved raw data and notebooks)"

