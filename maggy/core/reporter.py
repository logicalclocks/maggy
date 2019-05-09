"""
API Module for the user to include in his training code.

"""
import threading
from datetime import datetime

from maggy.core import exceptions, config

if config.mode is config.HOPSWORKS:
    from hops import hdfs

class Reporter(object):
    """
    Thread-safe store for sending a metric and logs from executor to driver
    """

    def __init__(self, log_file):
        self.metric = None
        self.lock = threading.RLock()
        self.stop = False
        self.trial_id = None
        self.logs = ''
        self.log_file = log_file

        #Open File desc for HDFS to log
        if config.mode is config.HOPSWORKS:
            if not hdfs.exists(log_file):
                hdfs.dump('', log_file)
            self.fd = hdfs.open_file(log_file, flags='w')

    # report
    def broadcast(self, metric):
        """Broadcast a metric to the experiment driver with the heartbeat.

        :param metric: Metric to be broadcasted
        :type metric: int, float
        :raises exception: EarlyStopException if told by the experiment driver
        """
        with self.lock:
            # if stop == True -> raise exception to break training function
            self.metric = metric
            if self.stop:
                raise exceptions.EarlyStopException(metric)

    def log(self, log_msg, print_executor=False):
        """Logs a message to the executor logfile and optionally to the
        executor sterr.

        :param log_msg: Message to log.
        :type log_msg: str
        :param print_executor: Print to executor sterr, defaults to False
        :type print_executor: bool, optional
        """
        with self.lock:
            msg = datetime.now().isoformat() + ': ' + str(log_msg)
            if print_executor:
                print(msg)
            if config.mode is config.HOPSWORKS:
                self.fd.write((msg + '\n').encode())
            self.logs = self.logs + msg + '\n'

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
