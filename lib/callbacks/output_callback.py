from abc import abstractmethod
from callbacks import Callback, Context

class OutputCallback(Callback):
    def __init__(self, each_n_epochs = 50): self.each_n_epochs = each_n_epochs

    def is_plot_time(self, ctx): return ctx.epoch() % self.each_n_epochs == 0 

    def can_plot(self, ctx):
        return (ctx.is_last_epoch() or (not ctx.is_first_epoch() and self.is_plot_time(ctx))) and ctx.verbose() > 0

    def on_after_train(self, ctx): 
        if self.can_plot(ctx): self.on_show(ctx)
    
    @abstractmethod
    def on_show(self, ctx: Context):
        pass