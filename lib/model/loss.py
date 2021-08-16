import torch.nn.functional as F
from utils import try_gpu
import torch
import numpy as np

def softmax_cross_entropy_fn(y_hat, y):
    # print(y, max_arg(y))
    y_max_arg = torch.max(y, 1)[1]
    return F.cross_entropy(y_hat, y_max_arg)


def softmax_pred_out(y_hat): return np.argmax(y_hat, axis=1)