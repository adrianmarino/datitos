from torch.optim import lr_scheduler
from callbacks import Callback

class ReduceLROnPlateau(Callback):
    def __init__(self, patience=10): self.patience = patience

    def on_init(self, args):
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            args['optimizer'],  
            'min', 
            patience = self.patience
        )

    def on_after_train(self, args):
        self.scheduler.step(args['val_loss'])