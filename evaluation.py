import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    classification_report,
    roc_auc_score,
    roc_curve
)
import pandas as pd
import os # Added to ensure that the save directory exists

# Directory to save plots
PLOT_DIR = "plots"

def ensure_plot_dir():
    """Ensures that the plots directory exists."""
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

def plot_confusion_matrix(y_true, y_pred, classes=None, title='Confusion Matrix', cmap=plt.cm.Blues, filename="confusion_matrix.png"):
    """Plots and saves the confusion matrix."""
    ensure_plot_dir() # Ensures the directory exists
    cm = confusion_matrix(y_true, y_pred)
    if classes is None:
        classes = ['Non-Fall', 'Fall'] # Assuming 0 and 1
    
    fig, ax = plt.subplots(figsize=(8, 6)) # Create figure and axes
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path) # Saves the figure
    print(f"Confusion matrix saved at: {save_path}")
    plt.close(fig) # Closes the figure to release memory

def plot_roc_curve(y_true, y_pred_proba, model_name="Model", filename="roc_curve.png"):
    """Plots and saves the ROC curve."""
    ensure_plot_dir() # Ensures the directory exists
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6)) # Create figure and axes
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')
    ax.plot([0, 1], [0, 1], 'k--') # Reference line (random)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True)
    
    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path) # Saves the figure
    print(f"ROC curve saved at: {save_path}")
    plt.close(fig) # Closes the figure to release memory

def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """Calculates and prints evaluation metrics and saves the graphs."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Evaluation Metrics for {model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"  ROC AUC:   {roc_auc:.4f}")
    
    print("\nClassification Report:")
    target_names = ['Non-Fall (0)', 'Fall (1)']
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    print("\nSaving Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, classes=target_names, filename=f"{model_name.replace(' ', '_')}_confusion_matrix.png")
    
    if y_pred_proba is not None:
        print("\nSaving ROC Curve...")
        plot_roc_curve(y_true, y_pred_proba, model_name=model_name, filename=f"{model_name.replace(' ', '_')}_roc_curve.png")

if __name__ == '__main__':
    print("This script defines evaluation functions. To test, run main.py.")
    # Example:
    # import numpy as np 
    # ensure_plot_dir()
    # y_true_sample = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0])
    # y_pred_sample = np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 0])
    # y_pred_proba_sample = np.array([0.1, 0.9, 0.2, 0.4, 0.3, 0.6, 0.8, 0.7, 0.4, 0.2]) # Prob of class 1
    # evaluate_model(y_true_sample, y_pred_sample, y_pred_proba_sample, model_name="Test RF") 