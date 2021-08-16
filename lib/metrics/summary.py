import numpy as np
from sklearn.metrics import classification_report
from metrics import show_score, plot_confusion_matrix, y_true_pred_values
from scikitplot.metrics import plot_roc

def show_summary(y_true, y_pred, y_pred_proba, cm_figsize=(6, 6), roc_figsize=(11, 8), labels=None):
    show_score(y_true, y_pred)

    print('Classification Report:')
    value_labels = [labels[r] for r in y_true_pred_values(y_true, y_pred)]
    print(classification_report(y_true, y_pred, target_names=value_labels))
    
    plot_confusion_matrix(y_true, y_pred, labels=labels, figsize=cm_figsize)
    
    plot_roc(y_true, y_pred_proba, figsize=roc_figsize)
