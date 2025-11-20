# Project Decisions and Assumptions

## Dataset Decisions

1. **Category Handling**: Used original category labels without modification
2. **Missing Values**: 
   - Product titles: Filled with empty string
   - Descriptions: Filled with empty string
   - No rows removed due to missing data

3. **Text Concatenation**: Combined title and description with space separator

## Preprocessing Decisions

1. **Number Retention**: Kept numbers in text (important for product specs like "64GB", "128GB")
2. **Stopword Removal**: Not applied (preserve semantic meaning in short product titles)
3. **Lemmatization**: Not applied (computational cost vs benefit trade-off)

## Feature Engineering Decisions

1. **TF-IDF Configuration**:
   - Max features: 50,000 (balance between coverage and dimensionality)
   - N-grams: (1,2) - unigrams and bigrams
   - Min DF: 2 (remove extremely rare terms)

2. **BERT Model Selection**: DistilBERT instead of BERT-base
   - Rationale: 40% smaller, 60% faster, 97% performance retention
   - Suitable for resource-constrained environments

## Training Decisions

1. **Random Seed**: Fixed at 42 for reproducibility
2. **Split Ratios**: 80/10/10 (standard practice for sufficient training data)
3. **Stratification**: Applied to preserve class distribution

4. **Hyperparameter Tuning**:
   - Used GridSearchCV (acceptable alternative to Optuna per assignment)
   - Limited parameter space for computational efficiency
   - Random Forest: Subsampled 25% of training data for Grid Search to reduce time

5. **BERT Training**:
   - Limited to 1 epoch for demonstration (resource constraints)
   - Batch size: 8 (memory limitations)
   - Max length: 128 tokens (shorter than standard 256 for efficiency)
   - Gradient accumulation: 2 (simulate larger batch size)
   - Disabled save_strategy="epoch" to avoid Windows file permission issues

## Evaluation Decisions

1. **Primary Metric**: Macro-F1 (handles class imbalance better than accuracy)
2. **Top-K Accuracy**: Computed for k=2,3,5 (relevant for e-commerce recommendations)
3. **ROC Curves**: Limited to top 10 categories by support (visualization clarity)

## Deployment Decisions

1. **Best Model**: Logistic Regression selected as baseline
   - Highest performance (96.92% test accuracy)
   - Fast inference
   - Low computational requirements
   
2. **Inference Interface**: CLI tool (simple, scriptable, testable)

## Deviations from Original Prompt

1. **LSTM Model**: Skipped (memory-intensive, lower priority than transformer model)
2. **Optuna**: Used GridSearchCV instead (equally valid per assignment specification)
3. **BERT Training**: Limited scope due to resource constraints
4. **API Development**: Created inference script; full API skeleton optional extension

## Technical Constraints

1. **Hardware**: CPU-only training (no GPU available)
2. **Memory**: Limited RAM necessitated batch size and sequence length reductions
3. **Time**: Project completion timeline required efficient choices

## Quality Assurance

1. **Code Quality**: Modular, documented, type-hinted where applicable
2. **Reproducibility**: Fixed random seeds, saved all artifacts, documented decisions
3. **Testing**: Inference script tested with sample predictions
4. **Documentation**: Comprehensive notebooks, final report, README

---

**Last Updated**: 2025-11-20
