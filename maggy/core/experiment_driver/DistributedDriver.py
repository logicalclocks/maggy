#
#   Copyright 2021 Logical Clocks AB
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

import threading
import queue

from pyspark.sql import SparkSession

from maggy.core.experiment_driver.Driver import Driver


class DistributedDriver(Driver):
    """Distributed driver class to run server in Torch registration mode.

    Attributes:
        server_addr (Union[str, None]): Address of the RPC server.
        job_start (Union[float, None]): Start time of the job.
        executor_addresses (dict): Contains the addresses and ports of the workers after successful
            reservation. Initially empty.
        spark_context (SparkContext): SparkContext of the current session.
    """

    def __init__(self, name, description, num_executors, hb_interval, log_dir):
        """Driver initialization.

        Args:
            name (str): Job name.
            description (str): Job description.
            num_executors (int): Number of Spark executors for the job.
            hb_interval (float): Heart beat time intervall.
            log_dir (str): Log directory path.
        """
        super().__init__(name, description, "max", num_executors, hb_interval, log_dir)
        self.server_addr = None
        self.num_trials = 1
        self.result = {"best_val": "n.a.", "num_trials": 1, "early_stopped": 0}
        self.job_start = None
        self.executor_addresses = {}
        self.spark_context = SparkSession.builder.getOrCreate().sparkContext

    def init(self, job_start):
        """Starts the server and worker to prepare for worker registration.

        Args:
            job_start (float): Job start time.
        """
        self.server_addr = self.server.start(self)
        self.job_start = job_start
        self._start_worker()

    def _start_worker(self):
        """Starts threaded worker to digest message queue.
        """

        def _digest_queue(self):
            try:
                while not self.worker_done:
                    try:
                        msg = self._message_q.get_nowait()
                    except queue.Empty:
                        msg = {"type": None}
                    if msg["type"] == "METRIC":
                        logs = msg.get("logs", None)
                        if logs is not None:
                            with self.log_lock:
                                self.executor_logs = self.executor_logs + logs
            except Exception as exc:  # pylint: disable=broad-except
                self._log(exc)
                self.exception = exc
                self.server.stop()
                raise

        threading.Thread(target=_digest_queue, args=(self,), daemon=True).start()

    def finalize(self, job_end):
        raise NotImplementedError

    def controller_get_next(self, trial=None):
        raise NotImplementedError

    def prep_results(self):
        pass

    def config_to_dict(self):
        raise NotImplementedError

    def log_string(self):
        pass
