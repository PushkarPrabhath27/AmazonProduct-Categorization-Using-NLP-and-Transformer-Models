**Module: Natural Language Processing**

**Assignment**

**_Assignment_**: Multi-Class Product Categorization using Transformer-based NLP Models

**_Problem Statement_**: In e-commerce platforms such as Amazon, Flipkart, and eBay, millions of products are listed under diverse categories. Manually labeling and categorizing these products based on their titles and descriptions is time-consuming and error-prone.The aim of the project is to build an AI-driven text classification system that can automatically predict the product category using Natural Language Processing (NLP) and Transformer-based models (e.g., BERT).

**_Data Sources:_**

Dataset Name: _Amazon Product Dataset (Public Kaggle version)  
_Data Source: Kaggle Dataset: Amazon Products Dataset  
Alternative public dataset: Amazon Product Review Data (UCSD)  
<br/>Data Fields Used:  
Column Description  
product_title Title of the product  
product_description Description text of the product  
category Product category label (target variable)  

**_Proposed Approach:_**

- Problem Understanding & Setup Define problem, import libraries, install dependencies
- Data Acquisition & Exploration Load dataset, handle missing values, EDA, visualize class distribution
- Data Preprocessing Clean text (lowercase, punctuation, stopwords), tokenize
- Text Representation Create TF-IDF & BERT embeddings for comparison
- Model Development Train and compare Logistic Regression, Random Forest, LSTM, and BERT models
- Model Evaluation & Optimization Fine-tune BERT, use hyperparameter optimization, evaluate metrics
- Model Saving & Deployment Prep Save final model, inference script, documentation
- Reporting & Insights Summarize results, visualize confusion matrix, prepare report

**_Methodology:_**

- Data Preprocessing
  - Remove HTML tags, punctuation, and numbers
  - Tokenization using nltk or transformers tokenizer
  - Stopword removal and lemmatization
- Text Vectorization
  - Compare traditional (TF-IDF) and deep (BERT embeddings) approaches
  - Use dimensionality reduction (PCA or TruncatedSVD) if necessary
- Model Building
  - Baseline: Logistic Regression, Naive Bayes
  - Advanced: Random Forest, LSTM
  - State-of-the-art: Fine-tuned bert-base-uncased using Hugging Face Transformers
- Model Evaluation
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix and ROC Curve visualization
- Optimization
  - Hyperparameter tuning using GridSearchCV or Optuna
  - Learning rate scheduling and early stopping (for deep learning models)
- Deployment Preparation
  - Save model using joblib or torch.save()
  - Create an inference function for real-time prediction

**_Expected Outcomes_**:

- Trained NLP model capable of classifying unseen product descriptions into relevant categories.
- Performance comparison between traditional ML and Transformer-based models.
- Final accuracy target: ≥ 85% for top categories.
- Reusable code pipeline for any multi-class text classification task.
- Documented insights for data imbalance, model interpretability, and deployment readiness.  

**_NOTE:_** Each of these steps should be accompanied by thorough documentation.

**References**

- Vaswani et al. (2017). Attention Is All You Need. <https://arxiv.org/abs/1706.03762>
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Kaggle Dataset - Amazon Product Dataset
- Hugging Face Transformers Documentation - <https://huggingface.co/docs>
- scikit-learn Documentation - <https://scikit-learn.org/stable/>
- Text preprocessing techniques - NLTK & spaCy official documentation.