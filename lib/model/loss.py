import torch.nn.functional as F
import torch
import numpy as np

def softmax_cross_entropy_fn(y_hat, y, reduction='mean'):
    y_max_arg = torch.max(y, 1)[1]
    return F.cross_entropy(y_hat, y_max_arg, reduction=reduction)

def softmax_pred_out(y_hat): return np.argmax(y_hat, axis=1)