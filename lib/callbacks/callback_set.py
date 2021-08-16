class CallbackSet:
    def __init__(self, callbacks=[]): self.callbacks = callbacks

    def on_init(self, model, optimizer, loss, verbose):
        context = { 
            'model': model,
            'optimizer': optimizer,
            'loss': loss,
            'verbose': verbose
        }
        [it.on_init(context) for it in self.callbacks]

    def on_after_train(self, model, optimizer, loss, verbose, epoch, train_set, val_set):
        lr = optimizer.param_groups[0]['lr']
        context = { 
            'model': model,
            'optimizer': optimizer,
            'epoch': epoch, 
            'loss': loss,
            'verbose': verbose,
            'lr': lr,
            'train_set': train_set, 
            'val_set': val_set
        }
        [it.on_after_train(context) for it in self.callbacks]