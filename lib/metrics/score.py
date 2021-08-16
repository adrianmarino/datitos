import numpy as np
from sklearn.metrics import accuracy_score

def show_score(y_true, y_pred):
    print('Accuracy: {:.4f} %\n'.format(accuracy_score(y_true, y_pred) * 100))

def y_true_pred_values(y_true, y_pred):
    true_values  = y_true[y_true.columns[0]].unique()            
    return np.sort(np.unique(np.concatenate((true_values, np.unique(y_pred)))))
