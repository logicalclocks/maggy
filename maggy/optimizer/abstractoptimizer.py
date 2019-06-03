from abc import ABC, abstractmethod


class AbstractOptimizer(ABC):

    def __init__(self):
        self.searchspace = None
        self.num_trials = None
        self.final_store = None

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_suggestion(self, trial=None):
        pass

    @abstractmethod
    def finalize_experiment(self, trials):
        pass
