import multiprocessing
from model.kfoldcv import KFoldCVStrategy, train_fold

class NonParallelKFoldCVStrategy(KFoldCVStrategy):
    def perform(self, folds, X, y, params, model_train_fn, callbacks):
        return [train_fold(model_train_fn, X, y, fold, train_idx, val_idx, params, callbacks) for fold, (train_idx, val_idx) in enumerate(folds)]