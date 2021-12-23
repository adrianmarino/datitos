import multiprocessing
from model.kfoldcv import KFoldCVStrategy, train_fold

class ParallelKFoldCVStrategy(KFoldCVStrategy):
    def __init__(self,  processes):
        self.__pool = multiprocessing.Pool(processes=processes)

    def perform(self, folds, X, y, params, model_train_fn, callbacks):
        params = [(model_train_fn, X, y, fold, train_idx, val_idx, params, callbacks) for fold, (train_idx, val_idx) in enumerate(folds)]
        return self.__pool.starmap(train_fold, params)