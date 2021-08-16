from torch.nn import Sequential, Linear, ReLU, Dropout

def linear_relu_dropout(n_inputs, n_output, dropout = 0.2):
    return Sequential(Linear(n_inputs, n_output), ReLU(), Dropout(dropout))
