from abc import ABC, abstractmethod


class AbstractModel(ABC):

    @abstractmethod
    def make_model(self):
        pass
