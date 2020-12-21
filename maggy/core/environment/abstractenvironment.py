



from abc import ABC, abstractmethod


class AbstractEnvironment(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def a_method(self):
        pass
