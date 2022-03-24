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
from typing import Callable, Type
import json
import os
import time
from tensorflow.keras import Model

from maggy import util
from maggy.core import config as mc

if mc.is_spark_available():
    from maggy.core.experiment_driver.spark_driver import Driver
else:
    from maggy.core.experiment_driver.python_driver import Driver
from maggy.core.executors.tf_dist_executor import dist_executor_fn
from maggy.config import TfDistributedConfig
from maggy.core.rpc import TensorflowServer
from maggy.core.environment.singleton import EnvSing


class TfDistributedTrainingDriver(Driver):
    """Driver for distributed learning with Tensorflow on a Spark cluster.

    Registers the workers on an RPC server, ensures proper configuration and
    logging, and accumulates final results.
    """

    def __init__(self, config: TfDistributedConfig, app_id: int, run_id: int):
        """Initializes the server, but does not start it yet.

        :param config: Experiment config.
        :param app_id: Maggy application ID.
        :param app_id: Maggy application ID.
        :param run_id: Maggy run ID.
        """
        super().__init__(config, app_id, run_id)
        self.data = []
        self.duration = None
        self.server = TensorflowServer(self.num_executors, config.__class__)
        self.final_model = None
        self.experiment_done = False
        self.result = []

    def set_result(self, result: float):
        self.result.append(result)

    def set_final_model(self, final_model: Model):
        self.final_model = final_model

    def finalize(self, job_end: float) -> dict:
        """Saves a summary of the experiment to a dict and logs it in the DFS.

        :param job_end: Time of the job end.

        :returns: The experiment summary dict.
        """
        self.job_end = job_end
        self.duration = util.seconds_to_milliseconds(self.job_end - self.job_start)
        duration_str = util.time_diff(self.job_start, self.job_end)
        results = self.prep_results(duration_str)
        print(results)
        self.log(results)
        EnvSing.get_instance().dump(
            json.dumps(self.result, default=util.json_default_numpy),
            self.log_dir + "/result.json",
        )
        EnvSing.get_instance().dump(self.json(), self.log_dir + "/maggy.json")
        return self.result

    def prep_results(self, duration_str: str) -> str:
        """Writes and returns the results of the experiment into one string and
        returns it.

        :param duration_str: Experiment duration as a formatted string.

        :returns: The formatted experiment results summary string.
        """
        results = (
            "Results ------\n"
            + "Metrics value "
            + str(self.result)
            + "\n"
            + "Total job time "
            + duration_str
            + "\n"
        )
        return results

    def json(self) -> str:
        """Exports the experiment's metadata in JSON format.

        :returns: The metadata string.
        """
        user = None
        constants = EnvSing.get_instance().get_constants()
        try:
            if constants.ENV_VARIABLES.HOPSWORKS_USER_ENV_VAR in os.environ:
                user = os.environ[constants.ENV_VARIABLES.HOPSWORKS_USER_ENV_VAR]
        except AttributeError:
            pass

        experiment_json = {
            "project": EnvSing.get_instance().project_name(),
            "user": user,
            "name": self.name,
            "module": "maggy",
            "app_id": str(self.app_id),
            "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.job_start)),
            "memory_per_executor": str(
                self.spark_context._conf.get("spark.executor.memory")
            ),
            "gpus_per_executor": str(
                self.spark_context._conf.get("spark.executor.gpus")
            ),
            "executors": self.num_executors,
            "logdir": self.log_dir,
            # 'versioned_resources': versioned_resources,
            "description": self.description,
        }

        if self.experiment_done:
            experiment_json["status"] = "FINISHED"
            experiment_json["finished"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self.job_end)
            )
            experiment_json["duration"] = self.duration
            experiment_json["config"] = json.dumps(self.final_model)
            experiment_json["metric"] = self.result

        else:
            experiment_json["status"] = "RUNNING"

        return json.dumps(experiment_json, default=util.json_default_numpy)

    def _exp_startup_callback(self) -> None:
        """No special startup actions required."""

    def _exp_final_callback(self, job_end: float, exp_json: dict) -> dict:
        """Calculates the average test error from all partitions.

        :param job_end: Time of the job end.
        :param _: Catches additional callback arguments.

        :returns: The result in a dictionary.
        """

        experiment_json = {"state": "FINISHED"}
        final_result = self.average_metric()

        util.finalize_experiment(
            exp_json,
            final_result,
            self.app_id,
            self.run_id,
            experiment_json,
            self.duration,
            self.log_dir,
            None,
            None,
        )
        self.experiment_done = True
        self.prep_results(str(job_end))

        print("Final average test loss: {:.3f}".format(final_result))
        print(
            "Finished experiment. Total run time: "
            + util.time_diff(self.job_start, job_end)
        )

        return {"test result": self.average_metric()}

    def _exp_exception_callback(self, exc: Type[Exception]) -> None:
        """Catches pickling errors in case either the model or the dataset are
        too large to be pickled, or not compatible.

        :param exc: The exception to handle.

        :raises RuntimeError: Provides the user with additional information
            about avoiding pickle problems and includes the pickle error.
        """
        self.experiment_done = True
        if isinstance(exc, PicklingError):
            raise RuntimeError(
                """Pickling has failed. This is most likely caused by one of the
                 following reasons: Model too large, model can't be pickled, dataset too large.
                 Consider passing a custom dataloader that reads from files in case of large
                 datasets or the model class instead of an instance. It will be initialized
                 automatically on the workers for you."""
            ) from exc
        raise exc

    def _patching_fn(
        self, train_fn: Callable, config: TfDistributedConfig, is_spark_available
    ) -> Callable:
        """Monkey patches the user training function with the distributed
        executor modifications for distributed training.

        :param train_fn: User provided training function.
        :param config: The configuration object for the experiment.

        :returns: The monkey patched training function."""

        return dist_executor_fn(
            train_fn,
            config,
            self.app_id,
            self.run_id,
            self.server_addr,
            self.hb_interval,
            self._secret,
            self.log_dir,
            is_spark_available,
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

        final_metric = msg.get("data", None)
        self.data.append(final_metric)
        self._update_result(final_metric)

    def _update_result(self, final_metric) -> None:
        self.result.append(final_metric)

    def average_metric(self) -> float:
        """Calculates the current average over the valid results.

        :returns: The average result value.
        """
        valid_results = [x for x in self.result if x is not None]
        if len(valid_results) > 0:
            return sum(valid_results) / len(valid_results)
        else:
            return 0
