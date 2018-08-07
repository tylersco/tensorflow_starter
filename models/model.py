'''
Adapted from https://danijar.com/structuring-your-tensorflow-models/
'''

from abc import ABC, abstractmethod

from .utils import AttrDict

class Model(ABC):

    def __init__(self):
        super().__init__()

    def get_config(self, config):
        c = AttrDict()
        for k, v in config.items():
            c[k] = v
        return c

    @abstractmethod
    def prediction(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def metrics(self):
        pass
