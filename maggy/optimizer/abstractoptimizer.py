from abc import ABC, abstractmethod


class AbstractOptimizer(ABC):

    def __init__(self):
        self.searchspace = None
        self.num_trials = None
        self.final_store = None

    @abstractmethod
    def initialize(self):
        """
        A hook for the developer to initialize the optimizer.
        """
        pass

    @abstractmethod
    def get_suggestion(self, trial=None):
        """
        Return a `Trial` to be assigned to an executor, or `None` if there are
        no trials remaining in the experiment.

        :rtype: `Trial` or `None`
        """
        pass

    @abstractmethod
    def finalize_experiment(self, trials):
        """
        This method will be called before finishing the experiment. Developers
        can implement this method e.g. for cleanup or extra logging.
        """
        pass
