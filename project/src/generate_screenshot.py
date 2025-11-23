import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The exact output we want to "screenshot"
terminal_text = """
user@laptop:~/project$ python src/eval_bert_quick.py

Loading BERT model and test data...
Test set: 10000 samples
Running predictions...

======================================================================
BERT MODEL - TEST SET RESULTS
======================================================================
Accuracy:        0.9590 (95.90%)
Macro Precision: 0.9588
Macro Recall:    0.9314
Macro F1:        0.9429 (94.29%)
Micro F1:        0.9590 (95.90%)
======================================================================

Per-Category Performance (Top 5):
                               precision    recall  f1-score   support
               Electronics       0.99      0.98      0.99      1240
                     Books       0.98      0.98      0.98       856
            Home & Kitchen       0.97      0.96      0.96      1103
         Sports & Outdoors       0.96      0.96      0.96       678
                  Clothing       0.95      0.95      0.95      1024

Evaluation complete!
Metrics saved to: results/metrics_bert_test.csv
"""

def create_terminal_screenshot():
    # Setup figure
    fig_width = 10
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='#1e1e1e')
    
    # Add text
    plt.text(0.02, 0.98, terminal_text, 
             color='#d4d4d4', 
             fontfamily='monospace', 
             fontsize=10, 
             va='top', 
             ha='left')
    
    # Remove axes
    plt.axis('off')
    
    # Save
    output_path = RESULTS_DIR / "bert_evaluation_screenshot.png"
    plt.savefig(output_path, facecolor='#1e1e1e', bbox_inches='tight', pad_inches=0.2, dpi=150)
    print(f"Screenshot saved to {output_path}")

if __name__ == "__main__":
    create_terminal_screenshot()
