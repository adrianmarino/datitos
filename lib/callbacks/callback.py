from abc import ABCMeta
from abc import abstractmethod

class Callback(metaclass=ABCMeta):
    def on_init(self, args):
        pass

    @abstractmethod
    def on_after_train(self, args):
        pass