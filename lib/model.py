from callbacks import CallbackSet
from data import InputDataIteratorFactory

class Model:
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def info(self):
        print('Model:\n', self.model)
        print('Params:\n', [(name, param.shape) for name, param in self.model.named_parameters()])
        return self

    def parameters(self): return self.model.parameters()

    def fit(
        self, 
        train_set, 
        val_set, 
        batch_size, 
        n_epochs, 
        verbose = 0, 
        callback_set = CallbackSet()
    ):
        callback_set.on_init(self.model, self.optimizer, self.loss, verbose)
        data_iter = InputDataIteratorFactory.create(train_set[0], train_set[1], batch_size)

        for epoch in range(n_epochs):
            for X, y in data_iter:
                l = self.loss(self.model(X), y)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

            callback_set.on_after_train(
                self.model, 
                self.optimizer, 
                self.loss, 
                verbose, 
                epoch, 
                train_set, 
                val_set
            )