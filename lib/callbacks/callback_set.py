class CallbackSet:
    def __init__(self, callbacks=[]): self.callbacks = callbacks

    def on_init(self, model, optimizer, metric, verbose):
        context = { 
            'model': model,
            'optimizer': optimizer,
            'metric': metric,
            'verbose': verbose
        }
        [it.on_init(context) for it in self.callbacks]

    def on_after_train(self, model, optimizer, metric, verbose, epoch, epochs, train_set, val_set):
        lr = optimizer.param_groups[0]['lr']
        context = { 
            'model': model,
            'optimizer': optimizer,
            'epochs': epochs,
            'epoch': epoch,
            'metric': metric,
            'verbose': verbose,
            'lr': lr,
            'train_set': train_set, 
            'val_set': val_set
        }
        [it.on_after_train(context) for it in self.callbacks]