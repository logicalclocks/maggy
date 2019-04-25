import statistics

from abc import ABC, abstractmethod


class AbstractEarlyStop(ABC):
    """An Abstract class to implement custom early stopping criteria."""

    @staticmethod
    @abstractmethod
    def earlystop_check(to_check, finalized_trials, direction):
        """A abstract static method that needs to be implemented with a custom
        early stopping criterium.

        The function is called internally in the user specified interval
        with three arguments. It is necessary to add these to the function
        definition.

        :param to_check: A dictionary of currently running
        trials, where the key is the `trial_id` and values are Trial objects.
        :type to_check: dictionary
        :param finalized_trials: A list of finalized Trial objects.
        :type finalized_trials: list
        :param direction: A string describing the search objective, i.e. 'min'
        or 'max'.
        :type direction: str
        """
        pass
