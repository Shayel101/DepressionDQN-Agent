# utils/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def compute_metrics(y_true, y_pred):
    """
    Compute ACC, Sensitivity (SEN), Specificity (SPE), and F1-Score.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    return {'ACC': acc, 'SEN': sensitivity, 'SPE': specificity, 'F1': f1}

if __name__ == '__main__':
    # Dummy test
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = compute_metrics(y_true, y_pred)
    print(metrics)
