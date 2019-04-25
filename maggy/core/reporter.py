"""
API Module for the user to include in his training code.

"""
import threading

from maggy import util

class Reporter(object):
    """
    Thread-safe store for sending a metric from executor to driver
    """

    def __init__(self):
        self.metric = None
        self.lock = threading.RLock()
        self.stop = False
        self.trial_id = None

    # report
    def broadcast(self, metric):

        with self.lock:
            # if stop == True -> raise exception to break training function
            self.metric = metric
            if self.stop:
                raise util.EarlyStopException(metric)

    def get_metric(self):

        with self.lock:
            return self.metric

    def reset(self):
        """
        Resets the reporter to the initial state in order to start a new
        experiment.
        """
        with self.lock:
            self.metric = None
            self.stop = False
            self.trial_id = None

    def early_stop(self):

        with self.lock:
            if self.metric is not None:
                self.stop = True

    def get_trial_id(self):
        with self.lock:
            return self.trial_id

    def set_trial_id(self, trial_id):
        with self.lock:
            self.trial_id = trial_id
