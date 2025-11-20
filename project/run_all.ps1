# PowerShell script to run the complete pipeline
# Usage: .\run_all.ps1

$ErrorActionPreference = "Stop"

Write-Host "Starting Amazon Product Categorization Pipeline" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Step 1: Data loading
Write-Host "`n[1/7] Loading and cleaning data..." -ForegroundColor Yellow
python src/data_loader.py
if ($LASTEXITCODE -ne 0) { throw "Data loading failed" }

# Step 2: Preprocessing
Write-Host "`n[2/7] Preprocessing data..." -ForegroundColor Yellow
python src/preprocess.py
if ($LASTEXITCODE -ne 0) { throw "Preprocessing failed" }

# Step 3: Feature engineering
Write-Host "`n[3/7] Generating features (TF-IDF and BERT embeddings)..." -ForegroundColor Yellow
python src/feature_engineering.py
if ($LASTEXITCODE -ne 0) { throw "Feature engineering failed" }

# Step 4: Train baseline models
Write-Host "`n[4/7] Training baseline models..." -ForegroundColor Yellow
python src/train_baselines.py --skip-lstm
if ($LASTEXITCODE -ne 0) { throw "Baseline training failed" }

# Step 5: Fine-tune BERT
Write-Host "`n[5/7] Fine-tuning BERT..." -ForegroundColor Yellow
python src/train_bert.py
if ($LASTEXITCODE -ne 0) { throw "BERT training failed" }

# Step 6: Evaluate models
Write-Host "`n[6/7] Evaluating models on test set..." -ForegroundColor Yellow
python src/eval.py
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }

# Step 7: Sample inference
Write-Host "`n[7/7] Running sample inference..." -ForegroundColor Yellow
python src/inference.py --title "Apple iPhone 12" --desc "64GB, black" --top-k 3 --model baseline
if ($LASTEXITCODE -ne 0) { Write-Warning "Inference test failed (non-critical)" }

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "Pipeline completed successfully!" -ForegroundColor Green
Write-Host "Check results/ for outputs and experiments/logs/ for logs" -ForegroundColor Cyan

