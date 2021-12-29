import logging
from abc import ABCMeta
from abc import abstractmethod

def train_fold(
    model_train_fn,
    X, y, 
    fold, 
    train_idx, val_idx, 
    params,
    callbacks
):
    score = model_train_fn(
        train_set = (X.iloc[train_idx], y.iloc[train_idx]),
        val_set   = (X.iloc[val_idx],   y.iloc[val_idx]),
        params    = params,
        callbacks = callbacks,
        fold      = fold
    )
    logging.info('Fold {} - Score: {}'.format(fold + 1, score))

    return score

class KFoldCVStrategy(metaclass=ABCMeta):
    @abstractmethod
    def perform(self, folds, X, y, params, model_train_fn, callbacks):
        pass