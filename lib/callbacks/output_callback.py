from abc import abstractmethod
from callbacks import Callback

class OutputCallback(Callback):
    def __init__(self, each_n_epochs = 50): self.each_n_epochs = each_n_epochs
    
    def can_plot(self, args):
        return args['epoch'] > 0 and args['epoch'] % self.each_n_epochs == 0 and args['verbose'] > 0

    def on_after_train(self, args): 
        if self.can_plot(args): self.on_show(args)
         
    @abstractmethod
    def on_show(self, model, optimizer, verbose, epoch, train_loss, val_loss):
        pass