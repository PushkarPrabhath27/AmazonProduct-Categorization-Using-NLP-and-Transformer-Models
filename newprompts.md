produce a runnable Google Colab notebook (instead of local scripts)
You are an autonomous ML/NLP engineer. Produce a single, fully runnable Google Colab notebook that implements the entire assignment step-by-step and saves artifacts to Google Drive. Work sequentially and do not skip steps. At the end of each major step, save outputs (files, models, logs) to a predictable folder in the user's Google Drive. Use GPU runtime when available. If any operation fails (OOM, missing dataset), attempt safe fallbacks (reduce batch size → use distilbert-base-uncased → use CPU) and record the fallback in NOTES.md in Drive.
Below are precise instructions for what the Colab notebook must contain and how it must behave. Generate the notebook programmatically (i.e., create notebook cells containing the code and markdown described below), save it automatically to the user's Google Drive (e.g., drive/MyDrive/product_categorization_colab.ipynb), and return a shareable link to the saved notebook (make it readable for the user). Do not ask the user for anything — assume the user will run the notebook themselves.
Colab runtime & Drive setup (first cells)
Add an initial markdown cell describing purpose and asking user to:
Set Colab runtime: Runtime → Change runtime type → GPU (or TPU if they prefer, but default GPU).
Optionally sign in to Kaggle (instructions and code cell provided) if dataset will be fetched via Kaggle API.
Add code cell to mount Google Drive:
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
PROJECT_ROOT = "/content/drive/MyDrive/product_categorization_project"
Create the folder structure inside Drive exactly as the original repo structure (create directories under PROJECT_ROOT).
Environment & dependencies (Colab cells)
Provide a code cell that installs required packages (use pip inside Colab). Pin versions appropriate for Colab + CUDA compatibility. Example cell:
%%bash
pip install --upgrade pip
pip install pandas==2.1.0 numpy==1.25.0 scikit-learn==1.3.2 nltk==3.8.1 spacy==3.5.0 transformers==4.40.0 datasets==2.15.0 torch==2.2.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html tqdm==4.66.1 seaborn==0.12.2 matplotlib==3.8.0 optuna==3.2.0 joblib==1.3.2 tensorboard==2.13.0 lime==0.2.0 shap==0.41.0
python -m nltk.downloader punkt stopwords wordnet
python -m spacy download en_core_web_sm
Save requirements.txt and environment.yml files into PROJECT_ROOT.
Dataset acquisition cell(s)
Provide two options (both cells with clear comments):
Option A: Upload a local CSV directly via Colab upload UI with code that moves the uploaded file to PROJECT_ROOT/data/raw/amazon_products.csv.
Option B: Download from Kaggle (provide kaggle.json upload instructions plus code to fetch dataset by dataset name / URL). Include clear fallback: if Kaggle credentials not provided, stop with an informative message telling user to upload CSV manually.
After obtaining the CSV, the notebook must verify columns and copy the assignment doc (if user uploaded the assignment doc to Drive) into PROJECT_ROOT/data/raw/ and record its path.
Notebook structure & required sections (cells)
Create separate, well-labeled notebook sections (markdown + code cells) matching Steps 1–12 from the assignment. Each section must include:

A short markdown description of the goals for that step.
Reproducible code cells that execute the step fully in Colab environment.
Save artifacts to Drive under PROJECT_ROOT after completion of the step.
Log short human-readable summaries to a summary.txt file in PROJECT_ROOT after each major step (append mode).
Specifically include the following sections and behaviors:

Step 1 — Problem understanding & setup (Colab)
Markdown summary of the problem (one paragraph) saved as PROJECT_ROOT/README.md.
Save NOTES.md with assumptions (multi-word categories, missing desc handling).
Save requirements.txt (from earlier cell) and a run_in_colab.sh that lists commands used.
Step 2 — Data acquisition & quick EDA (Colab)
Load CSV into a pandas DataFrame.
If dataset contains extra columns, merge/rename to ensure product_title, product_description, category.
Show head, shape, dtypes, unique category count, missing counts.
Generate and save PNGs into PROJECT_ROOT/results/plots/:
Top 20 categories bar plot.
Title length hist (tokens and chars).
Description length hist.
Class imbalance distribution (log scale if necessary).
Save an EDA notebook cell outputs snapshot by saving the executed notebook to Drive (the whole notebook will be saved at the end).
Step 3 — Data cleaning & preprocessing (Colab)
Implement reusable functions in the notebook (equivalent to src/preprocess.py), and save them as a standalone script in Drive PROJECT_ROOT/src/preprocess.py using Python %%bash + cat > technique (so user can later run scripts locally if needed).
Provide toggles (python variables) for cleaning flags: remove_numbers=True, apply_lemmatization=False by default.
Tokenization: show both nltk/spaCy route and Hugging Face tokenizer route.
Stratified split: use sklearn.model_selection.train_test_split with stratify and random_state=42 to produce train.csv, val.csv, test.csv saved in PROJECT_ROOT/data/processed/.
Step 4 — Feature representation (Colab)
Implement TF-IDF pipeline and save vectorizer (via joblib.dump) to PROJECT_ROOT/models/tfidf_vectorizer.joblib.
Implement BERT embedding extraction cell: load bert-base-uncased tokenizer + model, batch-encode and save [CLS] pooled vectors to PROJECT_ROOT/data/processed/embeddings/ as .npy. Provide size-check and fallback to distilbert if error/slow.
Save src/feature_engineering.py into Drive (use file write technique).
Step 5 — Baseline model training & evaluation (Colab)
Implement training cells for:
Logistic Regression (GridSearchCV) on TF-IDF features.
RandomForest on TF-IDF features.
MultinomialNB.
LSTM (use torch or keras — choose keras for Colab simplicity). If using keras, implement early stopping and class weighting.
Save best baseline model files to PROJECT_ROOT/models/ (.joblib for sklearn and .h5 for Keras).
Save baseline metrics CSV to PROJECT_ROOT/results/metrics_baselines.csv.
Step 6 — Fine-tune BERT (Colab)
Provide a runnable cell that fine-tunes with Hugging Face Trainer or transformers training loop.
Use the concatenation method (title + [SEP] + description) or tokenizer pair when building Dataset.
Include Optuna integration cell (or indicate where Optuna will run) — limit trials to e.g., 12 in Colab to be reasonable.
Save checkpoints and the best model and tokenizer to PROJECT_ROOT/models/bert_final/.
Save training logs to PROJECT_ROOT/experiments/logs/bert/ and enable TensorBoard output (provide a cell to launch %load_ext tensorboard).
Step 7 — Hyperparameter optimization (Colab)
Implement Optuna tuning for either baseline or BERT (user can switch via a boolean variable).
Save experiments/optuna_results.json and a plot of optimization trends.
Step 8 — Model evaluation on test set (Colab)
Load best models and run inference on test set.
Save confusion matrices (raw & normalized), classification report text, results/metrics.csv.
Compute top-2 and top-3 accuracy and save results.
If possible, compute one-vs-rest ROC curves and save PNGs.
Step 9 — Interpretability & error analysis (Colab)
Use LIME or SHAP (or both if available) on a sample of test examples (20 hard cases).
Save the LIME/SHAP visualizations and results into PROJECT_ROOT/results/interpretability/.
Save lists of 10 correct and 10 incorrect predictions with full metadata into CSV(s).
Step 10 — Model saving & deployment artifacts (Colab)
Save:
models/baseline.joblib
models/bert_final/ (model + tokenizer)
Export a standalone src/inference.py script into Drive that:
Loads model/tokenizer/vectorizer.
Exposes predict(title, description, top_k=3) and a CLI entrypoint.
Export a minimal FastAPI app file src/api/app.py into Drive.
Step 11 — Documentation & final report (Colab)
Render REPORT/final_report.md inside the notebook by programmatically assembling markdown text and saving it to Drive.
Use nbconvert to export the assembled notebook as REPORT/final_report.pdf into Drive.
Save notebooks/summary.ipynb in Drive (this can be a trimmed copy of the current notebook with only final evaluation cells).
Step 12 — Reproducibility checklist & delivery (Colab)
Create run_all_colab.sh (script with the high-level cell commands to re-run the pipeline in Colab).
Save config_colab.yaml with seeds, dataset paths, hyperparameters used.
Append a final summary.txt to PROJECT_ROOT listing final metrics (baseline and BERT) and paths to saved artifacts.
Colab-specific behaviors & constraints
All file operations must target PROJECT_ROOT in Drive so user retains artifacts after the session ends.
If GPU is available, automatically detect and print device info (torch.cuda.is_available()).
Where heavy compute is required (BERT fine-tuning), include a pre-check cell that estimates training time / memory and asks user to confirm by running the cell (but do not block creation of the notebook).
Limit long-running heavy experiments by default (e.g., num_train_epochs=3, per_device_train_batch_size=16), and clearly comment how to increase them.
Include try/except around heavy ops with fallback strategies documented and recorded in NOTES.md.
Notebook delivery requirements
When complete, the agent must:

Save the full Colab notebook to Drive at drive/MyDrive/product_categorization_project/product_categorization_colab.ipynb.
Save all scripts (src/*.py), models, results, and report files to the PROJECT_ROOT tree in Drive.
Print a final message in a notebook cell and log to PROJECT_ROOT/summary.txt containing:
The path to the saved notebook in Drive.
A shareable link to the notebook (or clear instructions for how the user can open it in Colab from Drive).
A short list of "how to run" steps (one-sentence each).
Provide a downloadable link or explicit instructions for the user to share the notebook from their Drive if the agent cannot set link permissions itself. (The notebook must include instructions for the user to right-click the file in Drive → Get link → set to "Anyone with the link" if they want to share).
Extra: UX & user guidance cells
Add a “Quick Start” cell at the top that — when executed — runs the minimal pipeline:
Mount Drive
Install requirements (skipped if already installed)
Preprocess data (if dataset exists)
Train a tiny baseline (e.g., TF-IDF + Logistic Regression on a 10k-subsample)
Evaluate and show metrics
This cell provides a quick demo run for users who don't want to run the full pipeline.
Acceptance criteria for the produced notebook
The notebook runs without modification in Colab (assuming user uploads the dataset or provides Kaggle credentials).
All artifacts are saved in Drive and their paths are printed at the end.
src/*.py files are written to Drive for later local re-use.
A final_report.pdf exists in Drive under REPORT/.
The notebook contains robust logging and fallbacks.
The notebook includes helpful comments and instructions so the user can re-run and modify experiments.
Final instruction to the agent
Generate the Colab notebook programmatically (cells with code and markdown) and save it to drive/MyDrive/product_categorization_project/product_categorization_colab.ipynb.
Also write the standalone Python scripts (src/*.py) into drive/MyDrive/product_categorization_project/src/.
At the end of the notebook, print the exact Drive paths for the notebook and key artifacts and save them to PROJECT_ROOT/summary.txt.
Do not ask the user any clarifying questions — assume dataset will be either uploaded or accessible via Kaggle and implement both code paths.
Make the notebook as user-friendly as possible: clear headings, short explanations before code cells, and an initial Quick Start runnable cell.



and the zip files are the datasets 