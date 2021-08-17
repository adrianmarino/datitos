class Context:
    def __init__(self, model, epochs, epoch, verbose,train_set, val_set):
        self.ctx = {
            'model'    : model,
            'epochs'   : epochs,
            'epoch'    : epoch,
            'verbose'  : verbose,
            'train_set': train_set,
            'val_set'  : val_set,
            'lr'       : model.optimizer.param_groups[0]['lr']    
        }

    def props(self):                 return self.ctx.items()
    def set_prop(self, name, value): self.ctx[name] = value
    def prop(self, name):            return self.ctx[name]
        
    def is_last_epoch(self):  return self.epochs() == self.epoch()
    def is_first_epoch(self): return self.epoch() == 1
    def epoch(self):          return self.prop('epoch')
    def epochs(self):         return self.prop('epochs')
    
    def model(self):          return self.prop('model')
    def wrapped_model(self):  return self.model().model

    def optimizer(self):      return self.model().optimizer
    def loss(self):           return self.model().loss

    def verbose(self):        return self.prop('verbose')
    
    def train_target(self):   return self.train_set()[1]
    def train_features(self): return self.train_set()[0]
    
    def val_target(self):     return self.val_set()[1]
    def val_features(self):   return self.val_set()[0]

    def train_set(self):     return self.prop('train_set')
    def val_set(self):       return self.prop('val_set')