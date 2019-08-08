"""
API Module for the user to include in his training code.

"""
import threading
from datetime import datetime

from maggy.core import exceptions, config

from hops import hdfs as hopshdfs

class Reporter(object):
    """
    Thread-safe store for sending a metric and logs from executor to driver
    """

    def __init__(self, log_file, partition_id, task_attempt, print_executor):
        self.metric = None
        self.lock = threading.RLock()
        self.stop = False
        self.trial_id = None
        self.logs = ''
        self.log_file = log_file
        self.partition_id = partition_id
        self.task_attempt = task_attempt
        self.print_executor = print_executor

        #Open File desc for HDFS to log
        if not hopshdfs.exists(log_file):
            hopshdfs.dump('', log_file)
        self.fd = hopshdfs.open_file(log_file, flags='w')

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

    def log(self, log_msg, verbose=True):
        """Logs a message to the executor logfile and executor stderr and
        optionally prints the message in jupyter.

        :param log_msg: Message to log.
        :type log_msg: str
        :param verbose: Print in Jupyter Notebook, defaults to True
        :type verbose: bool, optional
        """
        with self.lock:
            msg = datetime.now().isoformat() + \
                ' (' + str(self.partition_id) + '/' + \
                str(self.task_attempt) + '): ' + str(log_msg)
            self.fd.write((msg + '\n').encode())
            jupyter_log = str(self.partition_id) + ': ' + log_msg
            if verbose:
                self.logs = self.logs + jupyter_log + '\n'
            else:
                self.print_executor(msg)

    def get_data(self):
        """Returns the metric and logs to be sent to the experiment driver.
        """
        with self.lock:
            log_to_send = self.logs
            self.logs = ''
            return self.metric, log_to_send

    def reset(self):
        """
        Resets the reporter to the initial state in order to start a new
        trial.
        """
        with self.lock:
            self.metric = None
            self.stop = False
            self.trial_id = None
            self.fd.flush()

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
