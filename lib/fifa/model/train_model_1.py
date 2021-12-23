import numpy as np

from sklearn.metrics import balanced_accuracy_score

from callbacks import Logger, \
                      ReduceLROnPlateau, \
                      TrainValLossComputer, \
                      TrainValAccuracyComputer

from callbacks import CallbackSet
from metrics   import show_summary
from data      import to_single_col_df

from fifa.model import FifaModel1

def train_model_1(train_set, val_set, params, callbacks):
    model = FifaModel1(
        n_units        = [train_set[0].shape[1], params['hidden_units'], train_set[1].shape[1]],
        lr             = params['lr'],
        momentum       = params['momentum'],
        dropout        = params['dropout'],
        negative_slope = params['relu_neg_slope']
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

    return balanced_accuracy_score(y_true, y_pred)