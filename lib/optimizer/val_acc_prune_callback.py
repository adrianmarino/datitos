from optuna.exceptions import TrialPruned
from callbacks import Callback

class ValAccPruneCallback(Callback):
    def __init__(self, trial):
        self.trial = trial
    
    def on_after_train(self, ctx):
        accuracy = ctx.prop('val_acc')
        epoch    = ctx.prop('epoch')
    
        self.trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if self.trial.should_prune():
            raise TrialPruned()