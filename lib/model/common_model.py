from callbacks import CallbackSet
from data import InputDataIteratorFactory
from data import df_to_tensor

class CommonModel:
    def __init__(self, model, loss, optimizer, conv_pred_out_fn = lambda y: y):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.conv_pred_out_fn = conv_pred_out_fn

    def info(self):
        print('Model:\n', self.model)
        print('Params:\n', [(name, param.shape) for name, param in self.model.named_parameters()])
        return self
    
    def predict(self, X): 
        y_hat = self.model(df_to_tensor(X))
        y_hat = y_hat.cpu().detach().numpy()
        return self.conv_pred_out_fn(y_hat) 

    def fit(
        self, 
        train_set,
        val_set, 
        batch_size, 
        epochs,
        verbose = 1, 
        callback_set = CallbackSet()
    ):
        callback_set.on_init(self.model, self.optimizer, self.loss, verbose)
        data_iter = InputDataIteratorFactory.create(train_set[0], train_set[1], batch_size)

        for epoch in range(1, epochs+1):
            for X, y in data_iter:
                # Forward
                loss = self.loss(self.model(X), y)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            callback_set.on_after_train(
                self.model, 
                self.optimizer, 
                self.loss, 
                verbose, 
                epoch,
                epochs,
                train_set, 
                val_set
            )