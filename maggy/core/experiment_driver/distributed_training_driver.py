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

from pickle import PicklingError
from typing import Callable, Type, Any

from maggy import util
from maggy.experiment_config import TorchDistributedConfig
from maggy.core.rpc import DistributedTrainingServer
from maggy.core.experiment_driver.driver import Driver
from maggy.core.executors.torch_dist_executor import torch_dist_executor_fn


class DistributedTrainingDriver(Driver):
    """Driver for distributed learning on a Spark cluster.

    Registers the workers on an RPC server, ensures proper configuration and
    logging, and accumulates final results.
    """

    def __init__(self, config: TorchDistributedConfig, app_id: int, run_id: int):
        """Initializes the server for initial training setup communication and log collection.

        :param config: Experiment config.
        :param app_id: Maggy application ID.
        :param run_id: Maggy run ID.
        """
        super().__init__(config, app_id, run_id)
        self.server = DistributedTrainingServer(self.num_executors)
        self.results = []

    def _exp_startup_callback(self) -> None:
        """No special startup actions required."""

    def _exp_final_callback(self, job_end: float, _: Any) -> dict:
        """Calculates the average test error from all partitions.

        :param job_end: Time of the job end.
        :param _: Catches additional callback arguments.

        :returns: The result in a dictionary.
        """
        result = {"test result": self.average_metric()}
        print("Final average test loss: {:.3f}".format(self.average_metric()))
        print(
            "Finished experiment. Total run time: "
            + util.time_diff(self.job_start, job_end)
        )
        return result

    def _exp_exception_callback(self, exc: Type[Exception]) -> None:
        """Catches pickling errors in case either the model or the dataset are
        too large to be pickled, or not compatible.

        :param exc: The exception to handle.

        :raises RuntimeError: Provides the user with additional information
            about avoiding pickle problems and includes the pickle error.
        """
        if isinstance(exc, PicklingError):
            raise RuntimeError(
                """Pickling has failed. This is most likely caused by one of the
                 following reasons: Model too large, model can't be pickled, dataset too large.
                 Consider passing a custom dataloader that reads from files in case of large
                 datasets or the model class instead of an instance. It will be initialized
                 automatically on the workers for you."""
            )
        raise exc

    def _patching_fn(self, train_fn: Callable) -> Callable:
        """Monkey patches the user training function with the distributed
        executor modifications for distributed training.

        :param train_fn: User provided training function.

        :returns: The monkey patched training function.
        """
        return torch_dist_executor_fn(
            train_fn,
            self.config,
            self.app_id,
            self.run_id,
            self.server_addr,
            self.hb_interval,
            self._secret,
            self.log_dir,
        )

    def _register_msg_callbacks(self) -> None:
        """Registers a metric message callback for heartbeat responses to spark
        magic and a final callback to process experiment results.
        """
        self.message_callbacks["METRIC"] = self._log_msg_callback
        self.message_callbacks["FINAL"] = self._final_msg_callback

    def _log_msg_callback(self, msg: dict) -> None:
        """Callback for heartbeat messages with logs from the executors.

        :param msg: Message from the executors. Contains logs to be written to
            jupyter and the DFS.
        """
        logs = msg.get("logs", None)
        if logs is not None:
            with self.log_lock:
                self.executor_logs = self.executor_logs + logs

    def _final_msg_callback(self, msg: dict) -> None:
        """Appends the test result from the workers to the result list.

        :param msg: Final message from the executors.
        """
        self.results.append(msg.get("data", None))

    def average_metric(self) -> float:
        """Calculates the current average over the valid results.

        :returns: The average result value.
        """
        valid_results = [x for x in self.results if x is not None]
        return sum(valid_results) / len(valid_results)
