#!/bin/bash
# Bash script to run the complete pipeline
# Usage: bash run_all.sh

set -e  # Exit on error

echo "Starting Amazon Product Categorization Pipeline"
echo "================================================"

# Step 1: Data loading
echo ""
echo "[1/7] Loading and cleaning data..."
python src/data_loader.py

# Step 2: Preprocessing
echo ""
echo "[2/7] Preprocessing data..."
python src/preprocess.py

# Step 3: Feature engineering
echo ""
echo "[3/7] Generating features (TF-IDF and BERT embeddings)..."
python src/feature_engineering.py

# Step 4: Train baseline models
echo ""
echo "[4/7] Training baseline models..."
python src/train_baselines.py --skip-lstm

# Step 5: Fine-tune BERT
echo ""
echo "[5/7] Fine-tuning BERT..."
python src/train_bert.py

# Step 6: Evaluate models
echo ""
echo "[6/7] Evaluating models on test set..."
python src/eval.py

# Step 7: Sample inference
echo ""
echo "[7/7] Running sample inference..."
python src/inference.py --title "Apple iPhone 12" --desc "64GB, black" --top-k 3 --model baseline || echo "Inference test failed (non-critical)"

echo ""
echo "================================================"
echo "Pipeline completed successfully!"
echo "Check results/ for outputs and experiments/logs/ for logs"

