from data import df_to_tensor
from callbacks import Callback

class TrainValLossComputer(Callback):    
    def on_init(self, args): 
        self.__loss, self.__model = args['loss'], args['model']

    def __data_set_loss(self, df):
        return self.__loss(self.__model(df_to_tensor(df[0])), df_to_tensor(df[1])).item()

    def on_after_train(self, args):
        args['train_loss'] = self.__data_set_loss(args['train_set'])
        args['val_loss']   = self.__data_set_loss(args['val_set'])