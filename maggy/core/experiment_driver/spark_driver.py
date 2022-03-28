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

import time
import os
import queue
import threading
import secrets
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Tuple, Union

from maggy import util
from maggy.config import LagomConfig, BaseConfig
from maggy.core.environment.singleton import EnvSing
from maggy.config import (
    AblationConfig,
    HyperparameterOptConfig,
    TfDistributedConfig,
    TorchDistributedConfig,
)

DRIVER_SECRET = None


class Driver(ABC):
    """Abstract base driver class for the experiment drivers.

    The driver sets up a digestion thread for messages queued by the server and
    starts the experiment. Messages from the queue are used for communication
    between the Spark workers and the driver. Each derived experiment driver
    can register its own callbacks for custom message types. It should also
    implement the callbacks to customize the generic experiment setup to their
    needs.
    """

    SECRET_BYTES = 8

    def __init__(self, config: LagomConfig, app_id: int, run_id: int):
        """Sets up the RPC server, message queue and logs.

        :param config: Experiment config.
        :param app_id: Maggy application ID.
        :param run_id: Maggy run ID.
        """
        global DRIVER_SECRET
        self.config = config
        self.app_id = app_id
        self.run_id = run_id
        self.name = config.name
        self.description = config.description
        self.spark_context = util.find_spark().sparkContext
        self.num_executors = util.num_executors(self.spark_context)
        self.hb_interval = config.hb_interval
        self.server_addr = None
        self.job_start = None
        DRIVER_SECRET = (
            DRIVER_SECRET if DRIVER_SECRET else self._generate_secret(self.SECRET_BYTES)
        )
        self._secret = DRIVER_SECRET
        # Logging related initialization
        self._message_q = queue.Queue()
        self.message_callbacks = {}
        self._register_msg_callbacks()
        self.worker_done = False
        self.executor_logs = ""
        self.log_lock = threading.RLock()
        self.log_dir = EnvSing.get_instance().get_logdir(app_id, run_id)
        log_file = self.log_dir + "/maggy.log"
        # Open File desc for HDFS to log
        if not EnvSing.get_instance().exists(log_file):
            EnvSing.get_instance().dump("", log_file)
        self.log_file_handle = EnvSing.get_instance().open_file(log_file, flags="w")
        self.exception = None
        self.result = None
        self.result_dict = {}
        self.main_metric_key = None

    @staticmethod
    def _generate_secret(nbytes: int) -> str:
        """Generates a secret to be used by all clients during the experiment
        to authenticate their messages with the experiment driver.

        :param nbytes: Desired secret size.

        :returns: Secret string.
        """
        return secrets.token_hex(nbytes=nbytes)

    def run_experiment(
        self,
        train_fn: Callable,
        config: Union[
            AblationConfig,
            HyperparameterOptConfig,
            TfDistributedConfig,
            TorchDistributedConfig,
            BaseConfig,
        ],
    ) -> dict:
        """Runs the generic experiment setup with callbacks for customization.

        :param train_fn: User provided training function that should be
            parallelized.
        :param config: The configuration of the experiment.

        :returns: A dictionary of the experiment's results.
        """
        job_start = time.time()
        try:
            self._exp_startup_callback()
            exp_json = util.populate_experiment(
                self.config, self.app_id, self.run_id, str(self.__class__.__name__)
            )
            self.log(
                "Started Maggy Experiment: {}, {}, run {}".format(
                    self.name, self.app_id, self.run_id
                )
            )
            self.init(job_start)
            # Create a spark rdd partitioned into single integers, one for each executor. Allows
            # execution of functions on each executor node.
            node_rdd = self.spark_context.parallelize(
                range(self.num_executors), self.num_executors
            )
            self.spark_context.setJobGroup(
                os.environ["ML_ID"],
                "{} | {}".format(self.name, str(self.__class__.__name__)),
            )
            executor_fn = self._patching_fn(train_fn, config, True)
            # Trigger execution on Spark nodes.
            node_rdd.foreachPartition(executor_fn)

            job_end = time.time()
            result = self._exp_final_callback(job_end, exp_json)
            self._update_result(result)
            return result
        except Exception as exc:  # pylint: disable=broad-except
            self._exp_exception_callback(exc)
        finally:
            # Grace period to send last logs to sparkmagic.
            # Sparkmagic hb poll intervall is 5 seconds, therefore wait 6 seconds.
            time.sleep(6)
            self.stop()

    @abstractmethod
    def _exp_startup_callback(self) -> None:
        """Callback for experiment drivers to implement their own experiment
        startup logic.
        """

    @abstractmethod
    def _exp_final_callback(self, job_end: float, exp_json: dict) -> dict:
        """Callback for experiment drivers to implement their own experiment
        experiment finalization logic.

        :param job_end: Time of the job end.
        :param exp_json: Dictionary of experiment metadata.
        """

    @abstractmethod
    def _exp_exception_callback(self, exc: Exception):
        """Callback for experiment drivers to implement their own experiment
        error handling logic.

        :param exc: The caught exception.
        """

    @abstractmethod
    def _patching_fn(
        self,
        train_fn: Callable,
        config: Union[
            AblationConfig,
            HyperparameterOptConfig,
            TfDistributedConfig,
            TorchDistributedConfig,
            BaseConfig,
        ],
        is_spark_available: bool,
    ) -> Callable:
        """Patching function for the user provided training function.

        :param train_fn: User provided training function.

        :returns: The patched training function.
        """

    def init(self, job_start: float) -> None:
        """Starts the RPC server and message digestion worker.

        :param job_start: Time of the job start.
        """
        self.server_addr = self.server.start(self)
        self.job_start = job_start
        self._start_worker()

    def _start_worker(self) -> None:
        """Starts the message digestion worker thread.

        The worker tries to pop messages from the queue and matches their type
        keyword with any registered callbacks from the message_callback
        dictionary. The callback then gets called with the popped message.
        """

        def _digest_queue(self):
            try:
                while not self.worker_done:
                    try:
                        msg = self._message_q.get_nowait()
                    except queue.Empty:
                        msg = {"type": None}
                    if msg["type"] in self.message_callbacks.keys():
                        self.message_callbacks[msg["type"]](
                            msg
                        )  # Execute registered callbacks.
            except Exception as exc:  # pylint: disable=broad-except
                self.log(exc)
                self.exception = exc
                self.server.stop()
                raise

        threading.Thread(target=_digest_queue, args=(self,), daemon=True).start()

    @abstractmethod
    def _register_msg_callbacks(self) -> None:
        """Registers the callbacks for the message digestion thread.

        Since each driver can define its own message types and choose which
        ones to include, it has to be defined in the subclasses.
        """

    def add_message(self, msg: dict) -> None:
        """Adds a message to the message queue.

        :param msg: Message to put into the queue.
        """
        self._message_q.put(msg)

    def get_logs(self) -> Tuple[dict, str]:
        """Returns the current experiment status and executor logs to send them
        to spark magic.

        :returns: A tuple with the current experiment result and the aggregated
        executor log strings.
        """
        with self.log_lock:
            temp = self.executor_logs
            # clear the executor logs since they are being sent
            self.executor_logs = ""
            return self.result_dict[self.metric_key], temp

    def stop(self) -> None:
        """Stop the Driver's worker thread and server."""
        self.worker_done = True
        self.server.stop()
        self.log_file_handle.flush()
        self.log_file_handle.close()

    def log(self, log_msg: str) -> None:
        """Logs a string to the maggy driver log file.

        :param log_msg: The log message.
        """
        msg = datetime.now().isoformat() + ": " + str(log_msg)
        self.log_file_handle.write(EnvSing.get_instance().str_or_byte((msg + "\n")))

    @abstractmethod
    def _update_result(self, result) -> None:
        """Set the result variable.

        :param result: The value to set a result.
        """
