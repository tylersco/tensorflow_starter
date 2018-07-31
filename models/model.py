'''
Adapted from https://danijar.com/structuring-your-tensorflow-models/
'''

from abc import ABC, abstractmethod

class Model(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_config(self, config):
        pass

    @abstractmethod
    def prediction(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def metrics(self):
        pass
