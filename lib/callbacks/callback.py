from abc import ABCMeta
from abc import abstractmethod
from callbacks import Context

class Callback(metaclass=ABCMeta):
    def on_init(self, ctx: Context):
        pass

    @abstractmethod
    def on_after_train(self, ctx: Context):
        pass