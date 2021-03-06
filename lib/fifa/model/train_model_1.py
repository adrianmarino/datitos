import numpy as np
import random

from sklearn.metrics import balanced_accuracy_score

from callbacks import CallbackSet, \
                      Logger, \
                      ReduceLROnPlateau, \
                      TrainValLossComputer, \
                      TrainValAccuracyComputer

from metrics   import show_summary
from data      import to_single_col_df

from fifa.model import FifaModel1
from optuna.exceptions import TrialPruned

def train_model_1(train_set, val_set, params, callbacks, fold, trial = None):
    random.seed(params['seed'])

    units_per_layer = \
        [train_set[0].shape[1]] + \
        [params['hidden_units'] for _ in range(params['hidden_layers'])] + \
        [train_set[1].shape[1]]

    model = FifaModel1(
        units_per_layer = units_per_layer,
        lr              = params['lr'],
        momentum        = params['momentum'],
        dropout         = params['dropout'],
        negative_slope  = params['relu_neg_slope']
    )

    model.fit(
        train_set,
        val_set,
        batch_size   = params['batch_size'],
        epochs       = params['epochs'],
        callback_set = CallbackSet(
            [
                TrainValLossComputer(),
                TrainValAccuracyComputer(),
                Logger(metrics=['epoch', 'train_acc', 'val_acc'])
            ] + callbacks 
        )
    )

    y_true = to_single_col_df(np.argmax(val_set[1].values, axis=1))
    y_pred = to_single_col_df(model.predict(val_set[0]))

    accuracy = balanced_accuracy_score(y_true, y_pred)

    if trial:
        # Handle pruning based on the intermediate value.
        trial.report(accuracy, fold)
        if trial.should_prune():
            raise TrialPruned()

    return accuracy