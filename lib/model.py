from callbacks import CallbackSet
from data import InputDataIteratorFactory
from data import df_to_tensor

class Model:
    def __init__(self, model, metric, optimizer):
        self.model = model
        self.metric = metric
        self.optimizer = optimizer

    def info(self):
        print('Model:\n', self.model)
        print('Params:\n', [(name, param.shape) for name, param in self.model.named_parameters()])
        return self
    
    def predict(self, features): 
        target = self.model(df_to_tensor(features))
        return target.cpu().detach().numpy() 

    def fit(
        self, 
        train_set, 
        val_set, 
        batch_size, 
        epochs,
        verbose = 1, 
        callback_set = CallbackSet()
    ):
        callback_set.on_init(self.model, self.optimizer, self.metric, verbose)
        data_iter = InputDataIteratorFactory.create(train_set[0], train_set[1], batch_size)

        for epoch in range(1, epochs+1):
            for X, y in data_iter:
                # Forward
                metric = self.metric(self.model(X), y)

                # Backward
                self.optimizer.zero_grad()
                metric.backward()
                self.optimizer.step()

            callback_set.on_after_train(
                self.model, 
                self.optimizer, 
                self.metric, 
                verbose, 
                epoch,
                epochs,
                train_set, 
                val_set
            )