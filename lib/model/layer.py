from torch.nn import Sequential, Linear, LeakyReLU, Dropout
import numpy as np

def dense(n_inputs, n_output, act=LeakyReLU(), dropout = 0.2):
    return np.array([Linear(int(n_inputs), int(n_output)), act, Dropout(dropout)])

def dense_stack(units_per_layer, act=LeakyReLU(), dropout=0.2):
    """Create a stack of dense + dropot blocks. All dense layers have the same activation function and
    all dropout layes same probability to zeroes an element.

    Arguments:
        units_per_layer {[number]} -- The number of list elements is the number of layers. Each position define the number of units for this layers.

    Keyword Arguments:
        act {[Module]} -- Activation fuction (default: {LeakyReLU()})
        dropout {float} -- Probability to zeroes an element in dropout layers (default: {0.2})

    Returns:
        [Module] -- A list of stacked layers N * (Dense + Dropout).
    """
    return np.array([
        dense(units_per_layer[idx], units_per_layer[idx+1], act, dropout) for idx in range(0, len(units_per_layer)-1)
    ]).flatten()
