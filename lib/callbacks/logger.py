from callbacks import OutputCallback
import logging

class Logger(OutputCallback):
    def __init__(self, metrics=[], each_n_epochs = 50):
        super().__init__(each_n_epochs)
        self.metrics = metrics

    def on_show(self, ctx):
        logging.info({name:value for (name, value) in ctx.props() if name in self.metrics})
