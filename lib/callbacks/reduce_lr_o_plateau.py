from torch.optim import lr_scheduler
from callbacks import Callback

class ReduceLROnPlateau(Callback):
    def __init__(self, patience=10): self.patience = patience

    def on_init(self, ctx):
        self.scheduler = lr_scheduler.ReduceLROnPlateau(ctx.optimizer(), patience = self.patience)

    def on_after_train(self, ctx):
        self.scheduler.step(ctx.prop('val_loss'))