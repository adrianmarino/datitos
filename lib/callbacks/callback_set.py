from callbacks import Context 

class CallbackSet:
    def __init__(self, callbacks=[]):
        self.callbacks = callbacks

    def on_init(self, model, verbose):
        context = Context(model, None, None, verbose, None, None)
        [it.on_init(context) for it in self.callbacks]

    def on_after_train(self, model, verbose, epoch, epochs, train_set, val_set):
        context = Context(model, epochs, epoch, verbose, train_set, val_set)
        [it.on_after_train(context) for it in self.callbacks]