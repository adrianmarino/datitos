from callbacks import OutputCallback

class Logger(OutputCallback):
    def __init__(self, metrics=[], each_n_epochs = 50):
        super().__init__(each_n_epochs)
        self.metrics = metrics

    def on_show(self, args): 
        print({name:value for (name, value) in args.items() if name in self.metrics})
