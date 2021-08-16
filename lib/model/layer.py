from torch.nn import Sequential, Linear, ReLU, Dropout
import numpy as np

def dense(n_inputs, n_output, dropout = 0.2):
    return np.array([Linear(int(n_inputs), int(n_output)), ReLU(), Dropout(dropout)])

def dense_stack(n_units, dropout=0.2):
    return np.array([
        dense(n_units[idx], n_units[idx+1], dropout=dropout) for idx in range(0, len(n_units)-1)
    ]).flatten()
        

      