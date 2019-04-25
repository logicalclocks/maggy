from abc import ABC, abstractmethod

from maggy.trial import Trial

class AbstractOptimizer(ABC):

    def __init__(self, num_trials, searchspace):
        # Do stuff the user shouldn't see on initialization
        self.num_trials = num_trials
        self.searchspace = searchspace
        self.trial_buffer = []

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def get_suggestion(self, trial=None):
        pass

    @abstractmethod
    def finalize_experiment(self, trials):
        pass
