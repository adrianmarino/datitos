from callbacks import CallbackSet
from data import InputDataIteratorFactory
from data import df_to_tensor

class CommonModel:
    def __init__(self, model, loss, optimizer, conv_pred_out_fn = lambda y: y):
        """
        Requires a pytorch model, a loss, a optimizer and a funciont used to 
        convert model output (Optional). Last is used to convert mode output 
        to another outpt representation like a maxarg when last layer is a 
        softmax.
        """
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.conv_pred_out_fn = conv_pred_out_fn

    def info(self):
        print('Model:\n', self.model)
        print('Params:\n', [(name, param.shape) for name, param in self.model.named_parameters()])
        return self
    
    def predict(self, X):
        """
        Return model prediction.
        """
        y_hat = self.predict_proba(X)
        return self.conv_pred_out_fn(y_hat) 

    def predict_proba(self, X):
        """
        Return a probabiliry distribution for each input example.
        """
        self.__disable_regulatization_layers()        
        y_hat = self.model(df_to_tensor(X))
        self.__enable_regulatization_layers()
        return y_hat.cpu().detach().numpy()

    def __enable_regulatization_layers(self):
        """
        Activate regulatization layers (Dropout, BatchNorm, etc...)
        """
        self.model.train()

    def __disable_regulatization_layers(self):
        """
        Disable regulatization layers (Dropout, BatchNorm, etc...)
        """
        self.model.eval()
        
    def fit(
        self, 
        train_set,
        val_set, 
        batch_size, 
        epochs,
        verbose = 1, 
        callback_set = CallbackSet()
    ):
        """
        Perform train for the wrapped model. It peform coomon train steps like backpropagation, 
        weights update, and perform all specified callbacks after each batch train.
        """
        callback_set.on_init(self, verbose)
        data_iter = InputDataIteratorFactory.create(train_set[0], train_set[1], batch_size)

        for epoch in range(1, epochs+1):
            self.__enable_regulatization_layers()
            
            for X, y in data_iter:
                # Forward
                loss = self.loss(self.model(X), y)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()

                # Update model weights
                self.optimizer.step()

            self.__disable_regulatization_layers()

            callback_set.on_after_train(self, verbose, epoch, epochs, train_set, val_set)