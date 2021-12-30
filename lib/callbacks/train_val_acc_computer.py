import numpy as np
from data import df_to_tensor, to_single_col_df
from callbacks import Callback
from sklearn.metrics import balanced_accuracy_score

class TrainValAccuracyComputer(Callback):
    def on_after_train(self, ctx):
        ctx.set_prop('train_acc', self.compute_acc(ctx.train_target(), ctx.model().predict(ctx.train_features())))
        if ctx.val_set():
            ctx.set_prop('val_acc',   self.compute_acc(ctx.val_target(), ctx.model().predict(ctx.val_features())))

    def compute_acc(self, y_true, y_pred):
        y_true = to_single_col_df(np.argmax(y_true.values, axis=1))
        return balanced_accuracy_score(y_true, to_single_col_df(y_pred))