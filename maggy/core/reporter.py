#
#   Copyright 2020 Logical Clocks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

"""
API Module for the user to include in his training code.

"""
import threading
from datetime import datetime

from hops import hdfs as hopshdfs

from maggy import constants
from maggy.core import exceptions


class Reporter(object):
    """
    Thread-safe store for sending a metric and logs from executor to driver
    """

    def __init__(self, log_file, partition_id, task_attempt, print_executor):
        self.metric = None
        self.step = -1
        self.lock = threading.RLock()
        self.stop = False
        self.trial_id = None
        self.trial_log_file = None
        self.logs = ""
        self.log_file = log_file
        self.partition_id = partition_id
        self.task_attempt = task_attempt
        self.print_executor = print_executor

        # Open executor log file descriptor
        # This log is for all maggy system related log messages
        if not hopshdfs.exists(log_file):
            hopshdfs.dump("", log_file)
        self.fd = hopshdfs.open_file(log_file, flags="w")
        self.trial_fd = None

    def init_logger(self, trial_log_file):
        """Initializes the trial log file
        """
        self.trial_log_file = trial_log_file
        # Open trial log file descriptor
        if not hopshdfs.exists(self.trial_log_file):
            hopshdfs.dump("", self.trial_log_file)
        self.trial_fd = hopshdfs.open_file(self.trial_log_file, flags="w")

    def close_logger(self):
        """Savely closes the file descriptors of the log files.

        close() can be called multiple times and flushes the buffer contents
        before closing
        """
        with self.lock:
            if self.trial_fd:
                self.trial_fd.close()
            self.fd.close()

    # report
    def broadcast(self, metric, step=None):
        """Broadcast a metric to the experiment driver with the heartbeat.

        :param metric: Metric to be broadcasted
        :type metric: int, float
        :param step: The iteration step which produced the metric, e.g. batch or
            epoch number, or any other monotonically increasing progress attribute
        :type step: int
        :raises exception: EarlyStopException if told by the experiment driver
        """
        with self.lock:
            # if stop == True -> raise exception to break training function
            if step is None:
                step = self.step + 1
            if not isinstance(metric, constants.USER_FCT.NUMERIC_TYPES):
                raise exceptions.BroadcastMetricTypeError(metric)
            elif not isinstance(step, constants.USER_FCT.NUMERIC_TYPES):
                raise exceptions.BroadcastStepTypeError(metric, step)
            elif step < self.step:
                raise exceptions.BroadcastStepValueError(metric, step, self.step)
            else:
                self.step = step
                self.metric = metric
            if self.stop:
                raise exceptions.EarlyStopException(metric)

    def log(self, log_msg, jupyter=False):
        """Logs a message to the executor logfile and executor stderr and
        optionally prints the message in jupyter.

        :param log_msg: Message to log.
        :type log_msg: str
        :param verbose: Print in Jupyter Notebook, defaults to True
        :type verbose: bool, optional
        """
        with self.lock:
            try:
                msg = (datetime.now().isoformat() + " ({0}/{1}): {2} \n").format(
                    self.partition_id, self.task_attempt, log_msg
                )
                if jupyter:
                    jupyter_log = str(self.partition_id) + ": " + log_msg
                    self.trial_fd.write(msg.encode())
                    self.logs = self.logs + jupyter_log + "\n"
                else:
                    self.fd.write(msg.encode())
                    if self.trial_fd:
                        self.trial_fd.write(msg.encode())
                    self.print_executor(msg)
            # Throws ValueError when operating on closed HDFS file object
            # Throws AttributeError when calling file ops on NoneType object
            except (IOError, ValueError, AttributeError) as e:
                self.fd.write(
                    ("An error occurred while writing logs: {}".format(e)).encode()
                )

    def get_data(self):
        """Returns the metric and logs to be sent to the experiment driver.
        """
        with self.lock:
            log_to_send = self.logs
            self.logs = ""
            return self.metric, self.step, log_to_send

    def reset(self):
        """
        Resets the reporter to the initial state in order to start a new
        trial.
        """
        with self.lock:
            self.metric = None
            self.step = -1
            self.stop = False
            self.trial_id = None
            self.fd.flush()
            self.trial_fd.close()
            self.trial_fd = None
            self.trial_log_file = None

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
