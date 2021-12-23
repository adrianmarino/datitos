import logging
import multiprocessing
from sklearn.model_selection import StratifiedKFold
import numpy as np
from model.kfoldcv import NonParallelKFoldCVStrategy

class KFoldCV:
    def __init__(
        self, 
        model_train_fn,
        get_y_values_fn = lambda y: y.values.argmax(1),
        strategy        = NonParallelKFoldCVStrategy(), 
        k_fold          = 5,
        random_state    = 42,
        shuffle         = True,
        callbacks       = []
    ):
        self.__model_train_fn  = model_train_fn
        self.__get_y_values_fn = get_y_values_fn
        self.__strategy        = strategy
        self.__folder          = StratifiedKFold(
            n_splits=k_fold, 
            shuffle=shuffle, 
            random_state=random_state
        )
        self.__callbacks       = callbacks

    def train(self, X, y, params):
        np.random.seed(params['seed'])

        folds = self.__folder.split(X, self.__get_y_values_fn(y))

        scores = self.__strategy.perform(
            folds, 
            X, y, 
            params, 
            self.__model_train_fn, 
            self.__callbacks
        )

        return np.mean(scores)