from plot import plot_losses
from callbacks import OutputCallback
from IPython.display import clear_output

class PlotLosses(OutputCallback):
    def __init__(self, warmup_count = 0, each_n_epochs = 50): 
        super().__init__(each_n_epochs)
        self.train_losses, self.val_losses = [], []
        self.warmup_count  = warmup_count

    def on_after_train(self, args): 
        super().on_after_train(args)
        self.train_losses.append(args['train_loss'])
        self.val_losses.append(args['val_loss'])

    def on_show(self, args):
        if self.warmup_count < args['epoch']:
            clear_output(wait=True)
            plot_losses(self.train_losses, self.val_losses,  self.warmup_count)