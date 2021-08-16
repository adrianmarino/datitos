from plot import plot_losses
from callbacks import OutputCallback
from IPython.display import clear_output

class PlotLosses(OutputCallback):
    def __init__(self, warmup_count = 0, plot_each_n_epochs = 50, reg_each_n_epochs = 10): 
        super().__init__(plot_each_n_epochs)
        self.train_losses, self.val_losses, self.epochs = [], [], []
        self.warmup_count  = warmup_count
        self.reg_each_n_epochs = reg_each_n_epochs

    def on_after_train(self, args): 
        super().on_after_train(args)
        if args['epoch'] % self.reg_each_n_epochs == 0:
            self.train_losses.append(args['train_loss'])
            self.val_losses.append(args['val_loss'])
            self.epochs.append(args['epoch'])

    def on_show(self, args):
        clear_output(wait=True)
        plot_losses(self.train_losses, self.val_losses, self.epochs, self.warmup_count)