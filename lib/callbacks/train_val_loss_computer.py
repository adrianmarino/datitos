from data import df_to_tensor
from callbacks import Callback

class TrainValLossComputer(Callback):
    def compute_loss(self, ctx, df): return ctx.loss()(ctx.wrapped_model()(df_to_tensor(df[0])), df_to_tensor(df[1])).item()
    def on_after_train(self, ctx):
        ctx.set_prop('train_loss', self.compute_loss(ctx, ctx.train_set()))
        ctx.set_prop('val_loss',   self.compute_loss(ctx, ctx.val_set()))