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

from typing import Callable, Type

from maggy.core import config as mc

if mc.is_spark_available():
    from maggy.core.experiment_driver.spark_driver import Driver
else:
    from maggy.core.experiment_driver.python_driver import Driver
from maggy.core.executors.base_executor import base_executor_fn
from maggy.config import Config


class BaseDriver(Driver):
    """Driver for base experiments.

    Registers the workers on an RPC server, ensures proper configuration and
    logging, and accumulates final results.
    """

    def __init__(self, config: Config, app_id: int, run_id: int):
        """Initializes the server, but does not start it yet.

        :param config: Experiment config.
        :param app_id: Maggy application ID.
        :param app_id: Maggy application ID.
        :param run_id: Maggy run ID.
        """
        super().__init__(config, app_id, run_id)
        self.data = []
        self.duration = None
        self.server = None
        self.final_model = None
        self.experiment_done = False
        self.result = []

    def _exp_startup_callback(self) -> None:
        """No special startup actions required."""
        pass

    def _exp_final_callback(self, job_end: float, exp_json: dict) -> dict:
        """Calculates the average test error from all partitions."""
        pass

    def _exp_exception_callback(self, exc: Type[Exception]) -> None:
        """Catches pickling errors in case either the model or the dataset are
        too large to be pickled, or not compatible."""
        pass

    def _patching_fn(self, train_fn: Callable, config: Config) -> Callable:
        """Monkey patches the user training function with the distributed
        executor modifications for distributed training.

        :param train_fn: User provided training function.
        :param config: The configuration object for the experiment.

        :returns: The monkey patched training function."""

        return base_executor_fn(
            train_fn,
            config,
        )

    def _register_msg_callbacks(self) -> None:
        """Registers a metric message callback for heartbeat responses to spark
        magic and a final callback to process experiment results.
        """
        pass

    def run_experiment(
        self,
        train_fn: Callable,
        config: Config,
    ) -> dict:

        return train_fn()
