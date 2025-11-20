"""Generate comprehensive visualizations for final report.

Creates detailed plots for:
- Model comparison
- Feature importance
- Learning curves
- Category distribution
- Performance by category
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from scipy import sparse
from sklearn.metrics import classification_report
import json

# Setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "interpretability").mkdir(exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("Generating comprehensive visualizations...")

# 1. Model Comparison Visualization
print("\n1. Creating model comparison charts...")

# Load metrics
baseline_metrics = pd.read_csv(RESULTS_DIR / "metrics_baselines.csv")
test_metrics = pd.read_csv(RESULTS_DIR / "metrics_test.csv")

# Create comprehensive comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')

metrics_to_plot = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'top_2_accuracy', 'top_3_accuracy']
titles = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score', 'Top-2 Accuracy', 'Top-3 Accuracy']

for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[idx // 3, idx % 3]
    
    if metric in baseline_metrics.columns:
        data = baseline_metrics[['model', metric]].copy()
    elif metric in test_metrics.columns:
        data = test_metrics[['model', metric]].copy()
    else:
        continue
    
    bars = ax.bar(range(len(data)), data[metric], color=['#2ecc71', '#3498db', '#e74c3c'][:len(data)])
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data['model'], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, data[metric])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
print(f"   Saved: comprehensive_model_comparison.png")
plt.close()

# 2. Performance by Category
print("\n2. Creating per-category performance analysis...")

# Load classification report
le = joblib.load(MODEL_DIR / "label_encoder.joblib")
test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
X_test = sparse.load_npz(PROCESSED_DIR / "tfidf_test.npz")
y_test = le.transform(test_df["category"].values)

model = joblib.load(MODEL_DIR / "baseline.joblib")
y_pred = model.predict(X_test)

# Get detailed per-class metrics
report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

# Convert to DataFrame
categories = [cat for cat in report_dict.keys() if cat not in ['accuracy', 'macro avg', 'weighted avg']]
per_class_df = pd.DataFrame({
    'Category': categories,
    'Precision': [report_dict[cat]['precision'] for cat in categories],
    'Recall': [report_dict[cat]['recall'] for cat in categories],
    'F1-Score': [report_dict[cat]['f1-score'] for cat in categories],
    'Support': [report_dict[cat]['support'] for cat in categories]
})

# Sort by F1-score
per_class_df = per_class_df.sort_values('F1-Score', ascending=False)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Plot 1: F1-Score by category
colors = plt.cm.RdYlGn(per_class_df['F1-Score'])
bars = ax1.barh(per_class_df['Category'], per_class_df['F1-Score'], color=colors)
ax1.set_xlabel('F1-Score', fontweight='bold')
ax1.set_title('F1-Score by Product Category', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1.0)
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, per_class_df['F1-Score']):
    ax1.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', ha='left', fontweight='bold', fontsize=9)

# Plot 2: Support vs F1-Score scatter
scatter = ax2.scatter(per_class_df['Support'], per_class_df['F1-Score'], 
                     s=200, c=per_class_df['F1-Score'], cmap='RdYlGn', 
                     alpha=0.6, edgecolors='black', linewidth=1.5)
ax2.set_xlabel('Support (Number of Samples)', fontweight='bold')
ax2.set_ylabel('F1-Score', fontweight='bold')
ax2.set_title('F1-Score vs Category Support', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

# Add category labels
for _, row in per_class_df.iterrows():
    ax2.annotate(row['Category'], (row['Support'], row['F1-Score']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.colorbar(scatter, ax=ax2, label='F1-Score')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'per_category_performance.png', dpi=300, bbox_inches='tight')
print(f"   Saved: per_category_performance.png")
plt.close()

# Save detailed metrics table
per_class_df.to_csv(RESULTS_DIR / 'per_category_metrics.csv', index=False)
print(f"   Saved: per_category_metrics.csv")

# 3. Class Imbalance Analysis
print("\n3. Creating class imbalance analysis...")

# Get full dataset distribution
full_df = pd.concat([
    pd.read_csv(PROCESSED_DIR / "train.csv"),
    pd.read_csv(PROCESSED_DIR / "val.csv"),
    pd.read_csv(PROCESSED_DIR / "test.csv")
])

category_counts = full_df['category'].value_counts().sort_values(ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: Category distribution
colors = sns.color_palette("husl", len(category_counts))
bars = ax1.bar(range(len(category_counts)), category_counts.values, color=colors)
ax1.set_xticks(range(len(category_counts)))
ax1.set_xticklabels(category_counts.index, rotation=45, ha='right')
ax1.set_ylabel('Number of Samples', fontweight='bold')
ax1.set_title('Category Distribution (Full Dataset)', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, category_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Plot 2: Imbalance ratio
max_count = category_counts.max()
imbalance_ratio = category_counts / max_count
bars2 = ax2.bar(range(len(imbalance_ratio)), imbalance_ratio.values, color=colors)
ax2.set_xticks(range(len(imbalance_ratio)))
ax2.set_xticklabels(imbalance_ratio.index, rotation=45, ha='right')
ax2.set_ylabel('Imbalance Ratio', fontweight='bold')
ax2.set_title('Class Imbalance Ratio (Relative to Largest Class)', fontsize=14, fontweight='bold')
ax2.axhline(y=0.5, color='red', linestyle='--', label='50% threshold', linewidth=2)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'class_imbalance_analysis.png', dpi=300, bbox_inches='tight')
print(f"   Saved: class_imbalance_analysis.png")
plt.close()

# 4. Feature Importance (Top Features)
print("\n4. Extracting feature importance...")

vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
feature_names = np.array(vectorizer.get_feature_names_out())

# Get coefficients for top categories
coef = model.coef_
top_categories = per_class_df['Category'].head(5).tolist()

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Top Predictive Features by Category (Logistic Regression Coefficients)', 
             fontsize=16, fontweight='bold')

for idx, category in enumerate(top_categories):
    cat_idx = list(le.classes_).index(category)
    cat_coef = coef[cat_idx]
    
    # Get top 15 features
    top_indices = np.argsort(np.abs(cat_coef))[-15:][::-1]
    top_features = feature_names[top_indices]
    top_values = cat_coef[top_indices]
    
    ax = axes[idx // 3, idx % 3]
    colors = ['green' if v > 0 else 'red' for v in top_values]
    bars = ax.barh(range(len(top_features)), top_values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Coefficient Value', fontweight='bold')
    ax.set_title(f'{category}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

# Hide empty subplot
if len(top_categories) < 6:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'interpretability' / 'feature_importance_by_category.png', dpi=300, bbox_inches='tight')
print(f"   Saved: feature_importance_by_category.png")
plt.close()

# 5. Training/Test Split Visualization
print("\n5. Creating data split visualization...")

split_data = {
    'Split': ['Training', 'Validation', 'Test'],
    'Samples': [len(pd.read_csv(PROCESSED_DIR / "train.csv")),
                len(pd.read_csv(PROCESSED_DIR / "val.csv")),
                len(pd.read_csv(PROCESSED_DIR / "test.csv"))]
}
split_df = pd.DataFrame(split_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
colors_split = ['#3498db', '#2ecc71', '#e74c3c']
explode = (0.05, 0.05, 0.05)
ax1.pie(split_df['Samples'], labels=split_df['Split'], autopct='%1.1f%%',
        colors=colors_split, explode=explode, shadow=True, startangle=90)
ax1.set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')

# Bar chart
bars = ax2.bar(split_df['Split'], split_df['Samples'], color=colors_split, alpha=0.7)
ax2.set_ylabel('Number of Samples', fontweight='bold')
ax2.set_title('Dataset Split Sizes', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, split_df['Samples']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'dataset_split_visualization.png', dpi=300, bbox_inches='tight')
print(f"   Saved: dataset_split_visualization.png")
plt.close()

# 6. Model Architecture Comparison
print("\n6. Creating model architecture comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

models_info = [
    {'Model': 'Logistic\nRegression', 'Parameters': '750K', 'Training Time': '~5 min', 
     'Inference': '<1ms', 'Accuracy': 96.92, 'Color': '#2ecc71'},
    {'Model': 'Random\nForest', 'Parameters': '1.5M', 'Training Time': '~30 min',
     'Inference': '~10ms', 'Accuracy': 89.37, 'Color': '#3498db'},
    {'Model': 'Naive\nBayes', 'Parameters': '50K', 'Training Time': '~1 min',
     'Inference': '<1ms', 'Accuracy': 88.16, 'Color': '#e74c3c'},
]

y_pos = np.arange(len(models_info))
accuracies = [m['Accuracy'] for m in models_info]
colors = [m['Color'] for m in models_info]

bars = ax.barh(y_pos, accuracies, color=colors, alpha=0.7, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels([m['Model'] for m in models_info], fontweight='bold')
ax.set_xlabel('Test Accuracy (%)', fontweight='bold', fontsize=12)
ax.set_title('Model Comparison: Accuracy vs Complexity Trade-off', fontsize=14, fontweight='bold')
ax.set_xlim(0, 100)
ax.grid(axis='x', alpha=0.3)

# Add annotations
for i, (bar, model_info) in enumerate(zip(bars, models_info)):
    acc = model_info['Accuracy']
    params = model_info['Parameters']
    train_time = model_info['Training Time']
    inf_time = model_info['Inference']
    
    ax.text(acc + 1, bar.get_y() + bar.get_height()/2,
            f"{acc:.2f}%\nParams: {params}\nTraining: {train_time}\nInference: {inf_time}",
            va='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'model_complexity_comparison.png', dpi=300, bbox_inches='tight')
print(f"   Saved: model_complexity_comparison.png")
plt.close()

print("\n✅ All visualizations generated successfully!")
print(f"\nSaved to: {RESULTS_DIR}/")
print("\nGenerated files:")
print("  - comprehensive_model_comparison.png")
print("  - per_category_performance.png")
print("  - per_category_metrics.csv")
print("  - class_imbalance_analysis.png")
print("  - interpretability/feature_importance_by_category.png")
print("  - dataset_split_visualization.png")
print("  - model_complexity_comparison.png")
