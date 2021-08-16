from abc import abstractmethod
from callbacks import Callback

class OutputCallback(Callback):
    def __init__(self, each_n_epochs = 50): self.each_n_epochs = each_n_epochs

    def is_last(self, args):      return args['epochs'] == args['epoch']
    def is_first(self, args):     return args['epoch'] == 1
    def is_plot_time(self, args): return args['epoch'] % self.each_n_epochs == 0    
    def can_plot(self, args):
        return (self.is_last(args) or (not self.is_first(args) and self.is_plot_time(args))) and args['verbose'] > 0

    def on_after_train(self, args): 
        if self.can_plot(args): self.on_show(args)
         
    @abstractmethod
    def on_show(self, model, optimizer, verbose, epoch, train_loss, val_loss):
        pass