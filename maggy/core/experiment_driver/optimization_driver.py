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

import os
import time
import json
from typing import Callable, Optional, Union

from maggy import util, tensorboard
from maggy.searchspace import Searchspace
from maggy.optimizer import AbstractOptimizer, RandomSearch, Asha, SingleRun, GridSearch
from maggy.earlystop import AbstractEarlyStop, MedianStoppingRule, NoStoppingRule
from maggy.optimizer import bayes
from maggy.trial import Trial
from maggy.core import config as mc

if mc.is_spark_available():
    from maggy.core.experiment_driver.spark_driver import Driver
else:
    from maggy.core.experiment_driver.python_driver import Driver
from maggy.core.rpc import OptimizationServer
from maggy.core.environment.singleton import EnvSing
from maggy.core.executors.trial_executor import trial_executor_fn
from maggy.config import AblationConfig, HyperparameterOptConfig


class HyperparameterOptDriver(Driver):
    """Driver class for hyperparameter optimization experiments.

    Initializes a controller that returns a new hyperparameter configuration
    from the searchspace on each poll and sets up the callbacks for the hp
    optimization.
    """

    # When adding a controller_dict entry, make sure the key is in lower case
    controller_dict = {
        "randomsearch": RandomSearch,
        "asha": Asha,
        "tpe": bayes.TPE,
        "gp": bayes.GP,
        "none": SingleRun,
        "faulty_none": None,
        "gridsearch": GridSearch,
    }

    def __init__(self, config: HyperparameterOptConfig, app_id: int, run_id: int):
        """Performs argument checks and initializes the optimization
        controller.

        :param config: Experiment config.
        :param app_id: Maggy application ID.
        :param run_id: Maggy run ID.

        :raises ValueError: In case an invalid optimization direction was
            specified.
        """
        super().__init__(config, app_id, run_id)
        self._final_store = []
        self._trial_store = {}
        self.experiment_done = False
        self.maggy_log = ""
        self.job_end = None
        self.duration = None
        # Interrupt init for AblationDriver.
        if isinstance(config, AblationConfig):
            return
        self.num_trials = config.num_trials
        self.num_executors = min(
            util.num_executors(self.spark_context), self.num_trials
        )
        self.server = OptimizationServer(self.num_executors, config.__class__)
        self.searchspace = self._init_searchspace(config.searchspace)
        self.controller = self._init_controller(config.optimizer, self.searchspace)
        # if optimizer has pruner, num trials is determined by pruner
        if self.controller.pruner:
            self.num_trials = self.controller.pruner.num_trials()

        if isinstance(self.controller, GridSearch):
            # number of trials need to be determined depending on searchspace of user.
            self.num_trials = self.controller.get_num_trials(config.searchspace)

        self.earlystop_check = self._init_earlystop_check(config.es_policy)
        self.es_interval = config.es_interval
        self.es_min = config.es_min
        if isinstance(config.direction, str) and config.direction.lower() in [
            "min",
            "max",
        ]:
            self.direction = config.direction.lower()
        else:
            raise ValueError(
                "The experiment's direction should be a string (either 'min' or 'max') "
                "but it is {0} (of type '{1}').".format(
                    str(config.direction), type(config.direction).__name__
                )
            )
        self.result = {"best_val": "n.a.", "num_trials": 0, "early_stopped": 0}
        # Init controller and set references to data
        self.controller.num_trials = self.num_trials
        self.controller.searchspace = self.searchspace
        self.controller.trial_store = self._trial_store
        self.controller.final_store = self._final_store
        self.controller.direction = self.direction
        self.controller._initialize(exp_dir=self.log_dir)

    def _exp_startup_callback(self) -> None:
        """Registers the hp config to tensorboard upon experiment startup."""
        tensorboard._register(
            EnvSing.get_instance().get_logdir(self.app_id, self.run_id)
        )
        tensorboard._write_hparams_config(
            EnvSing.get_instance().get_logdir(self.app_id, self.run_id),
            self.config.searchspace,
        )

    def _exp_final_callback(self, job_end: float, exp_json: dict) -> dict:
        """Writes the results from the hp optimization into a dict and logs it.

        :param job_end: Time of the job end.
        :param exp_json: Dictionary of experiment metadata.

        :returns: A summary of the ablation study results.
        """
        result = self.finalize(job_end)
        best_logdir = self.log_dir + "/" + result["best_id"]
        util.finalize_experiment(
            exp_json,
            float(result["best_val"]),
            self.app_id,
            self.run_id,
            "FINISHED",
            self.duration,
            self.log_dir,
            best_logdir,
            self.config.optimization_key,
        )
        print("Finished experiment.")
        return result

    def _exp_exception_callback(self, exc: Exception) -> None:
        """Closes logs, raises the driver exception if existent, else reraises
        unhandled exception.

        :param exc: The exception to handle.
        """
        self.controller._close_log()
        if self.controller.pruner:
            self.controller.pruner._close_log()
        if self.exception:
            raise self.exception  # pylint: disable=raising-bad-type
        raise exc

    def _patching_fn(
        self,
        train_fn: Callable,
        config: HyperparameterOptConfig,
        is_spark_available: bool,
    ) -> Callable:
        """Monkey patches the user training function with the trial executor
        modifications for hyperparameter search.


        :param train_fn: User provided training function.
        :param config: The configuration object for the experiment.

        :returns: The monkey patched training function."""
        return trial_executor_fn(
            train_fn,
            config,
            "optimization",
            self.app_id,
            self.run_id,
            self.server_addr,
            self.hb_interval,
            self._secret,
            self.config.optimization_key,
            self.log_dir,
        )

    def _register_msg_callbacks(self) -> None:
        """Registers message callbacks for heartbeat responses to spark
        magic, blacklist messages to exclude hp configurations, final callbacks
        to process experiment results, idle callbacks for finished executors,
        and registration callbacks for the clients to exchange connection info.
        """
        for key, call in (
            ("METRIC", self._metric_msg_callback),
            ("BLACK", self._blacklist_msg_callback),
            ("FINAL", self._final_msg_callback),
            ("IDLE", self._idle_msg_callback),
            ("REG", self._register_msg_callback),
        ):
            self.message_callbacks[key] = call

    def controller_get_next(self, trial: Optional[Trial] = None) -> Union[Trial, None]:
        """Gets a `Trial` to be assigned to an executor, or `None` if there are
        no trials remaining in the experiment.

        :param trial: Trial to fetch from the controller (default ``None``).
            None autofetches the next available trial.

        :returns: A new trial for hp optimization.
        """
        return self.controller.get_suggestion(trial)

    def get_trial(self, trial_id: int) -> Trial:
        """Returns a trial by ID from the trial store.

        :param trial_id: The target trial ID.

        :returns: The trial configuration.
        """
        return self._trial_store[trial_id]

    def add_trial(self, trial: Trial) -> None:
        """Adds a trial to the internal trial store.

        :param trial: The trial to be added.
        """
        self._trial_store[trial.trial_id] = trial

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
        self.controller._finalize_experiment(self._final_store)
        results = (
            "\n------ "
            + self.controller.name()
            + " Results ------ direction("
            + self.direction
            + ") \n"
            "BEST combination "
            + json.dumps(self.result["best_config"])
            + " -- metric "
            + str(self.result["best_val"])
            + "\n"
            "WORST combination "
            + json.dumps(self.result["worst_config"])
            + " -- metric "
            + str(self.result["worst_val"])
            + "\n"
            "AVERAGE metric -- " + str(self.result["avg"]) + "\n"
            "EARLY STOPPED Trials -- " + str(self.result["early_stopped"]) + "\n"
            "Total job time " + duration_str + "\n"
        )
        return results

    def config_to_dict(self) -> dict:
        """Returns a summary of the scheduled hp optimization searchspace as a
            dict.

        :returns: The summary dict.
        """
        return self.searchspace.to_dict()

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
            "experiment_type": self.controller.name(),
        }

        experiment_json["controller"] = self.controller.name()
        experiment_json["config"] = json.dumps(self.config_to_dict())

        if self.experiment_done:
            experiment_json["status"] = "FINISHED"
            experiment_json["finished"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self.job_end)
            )
            experiment_json["duration"] = self.duration
            experiment_json["config"] = json.dumps(self.result["best_config"])
            experiment_json["metric"] = self.result["best_val"]

        else:
            experiment_json["status"] = "RUNNING"

        return json.dumps(experiment_json, default=util.json_default_numpy)

    def _update_result(self, trial: Trial) -> None:
        """Updates the current result's best and worst trial given a finalized
            trial.

        :param trial: The finalized trial.
        """
        if isinstance(trial, Trial):
            metric = trial.final_metric
            param_string = trial.params
            trial_id = trial.trial_id
            num_epochs = len(trial.metric_history)

            # pop function values and trial_type from parameters, since we don't need them
            param_string.pop("dataset_function", None)
            param_string.pop("model_function", None)
            # First finalized trial
            if self.result.get("best_id", None) is None:
                self.result = {
                    "best_id": trial_id,
                    "best_val": metric,
                    "best_config": param_string,
                    "worst_id": trial_id,
                    "worst_val": metric,
                    "worst_config": param_string,
                    "avg": metric,
                    "metric_list": [metric],
                    "num_trials": 1,
                    "early_stopped": 0,
                    "num_epochs": num_epochs,
                    "trial_id": trial_id,
                }
                if trial.early_stop:
                    self.result["early_stopped"] += 1
                return
            else:
                if self.direction == "max":
                    if metric > self.result["best_val"]:
                        self.result["best_val"] = metric
                        self.result["best_id"] = trial_id
                        self.result["best_config"] = param_string
                    if metric < self.result["worst_val"]:
                        self.result["worst_val"] = metric
                        self.result["worst_id"] = trial_id
                        self.result["worst_config"] = param_string
                elif self.direction == "min":
                    if metric < self.result["best_val"]:
                        self.result["best_val"] = metric
                        self.result["best_id"] = trial_id
                        self.result["best_config"] = param_string
                    if metric > self.result["worst_val"]:
                        self.result["worst_val"] = metric
                        self.result["worst_id"] = trial_id
                        self.result["worst_config"] = param_string

            # update results and average regardless of experiment type
            self.result["metric_list"].append(metric)
            self.result["num_trials"] += 1
            self.result["avg"] = sum(self.result["metric_list"]) / float(
                len(self.result["metric_list"])
            )

            if trial.early_stop:
                self.result["early_stopped"] += 1

    def _update_maggy_log(self) -> None:
        """Creates the status of a maggy experiment with a progress bar."""
        return self.log_string()

    def log_string(self) -> str:
        """Creates a summary of the current experiment progress for jupyter.

        :returns: The summary string.
        """
        log = (
            "Maggy Optimization "
            + str(self.result["num_trials"])
            + "/"
            + str(self.num_trials)
            + " ("
            + str(self.result["early_stopped"])
            + ") "
            + util.progress_bar(self.result["num_trials"], self.num_trials)
            + " - BEST "
            + json.dumps(self.result["best_config"])
            + " - metric "
            + str(self.result["best_val"])
        )
        return log

    def _metric_msg_callback(self, msg: dict) -> None:
        """Heartbeart message callback.

        Checks if the experiment is underperforming and can trigger an early
        stop abort. Also copies logs from the server to the driver logs for
        later display in sparkmagic.

        :param msg: The metric message from the message queue.
        """
        logs = msg.get("logs", None)
        if logs is not None:
            with self.log_lock:
                self.executor_logs = self.executor_logs + logs

        step = None
        if msg["trial_id"] is not None and msg["data"] is not None:
            step = self.get_trial(msg["trial_id"]).append_metric(msg["data"])

        # maybe these if statements should be in a function
        # also this could be made a separate message
        # i.e. step nr is added to the queue as message which will
        # then later be checked for early stopping, just to not
        # block for too long for other messages
        if self.earlystop_check != NoStoppingRule.earlystop_check:
            if len(self._final_store) > self.es_min:
                if step is not None and step != 0:
                    if step % self.es_interval == 0:
                        try:
                            to_stop = self.earlystop_check(
                                self.get_trial(msg["trial_id"]),
                                self._final_store,
                                self.direction,
                            )
                        except Exception as e:
                            self.log(e)
                            to_stop = None
                        if to_stop is not None:
                            self.log("Trials to stop: {}".format(to_stop))
                            self.get_trial(to_stop).set_early_stop()

    def _blacklist_msg_callback(self, msg: dict) -> None:
        """Blacklist message callback.

        Registers a trial in the server reservations.

        :param msg: The blacklist message from the message queue.
        """
        trial = self.get_trial(msg["trial_id"])
        with trial.lock:
            trial.status = Trial.SCHEDULED
            self.server.reservations.assign_trial(msg["partition_id"], msg["trial_id"])

    def _final_msg_callback(self, msg: dict) -> None:
        """Final message callback.

        Logs trial results and registers executor as idle.

        :param msg: The final executor message from the message queue.
        """
        trial = self.get_trial(msg["trial_id"])
        logs = msg.get("logs", None)
        if logs is not None:
            with self.log_lock:
                self.executor_logs = self.executor_logs + logs

        # finalize the trial object
        with trial.lock:
            trial.status = Trial.FINALIZED
            trial.final_metric = msg["data"]
            trial.duration = util.seconds_to_milliseconds(time.time() - trial.start)

        # move trial to the finalized ones
        self._final_store.append(trial)
        self._trial_store.pop(trial.trial_id)

        # update result dictionary
        self._update_result(trial)
        # keep for later in case tqdm doesn't work
        self.maggy_log = self._update_maggy_log()
        self.log(self.maggy_log)

        EnvSing.get_instance().dump(
            trial.to_json(),
            self.log_dir + "/" + trial.trial_id + "/trial.json",
        )

        # assign new trial
        trial = self.controller_get_next(trial)
        if trial is None:
            self.server.reservations.assign_trial(msg["partition_id"], None)
            self.experiment_done = True
        elif trial == "IDLE":
            self.add_message(
                {
                    "type": "IDLE",
                    "partition_id": msg["partition_id"],
                    "idle_start": time.time(),
                }
            )
            self.server.reservations.assign_trial(msg["partition_id"], None)
        else:
            with trial.lock:
                trial.start = time.time()
                trial.status = Trial.SCHEDULED
                self.server.reservations.assign_trial(
                    msg["partition_id"], trial.trial_id
                )
                self.add_trial(trial)

    def _idle_msg_callback(self, msg: dict) -> None:
        """Idle message callback.

        Tries to trigger another trial for the idle executor.

        :param msg: The idle message from the message queue.
        """
        # execute only every 0.1 seconds but do not block thread
        if time.time() - msg["idle_start"] > 0.1:
            trial = self.controller_get_next()
            if trial is None:
                self.server.reservations.assign_trial(msg["partition_id"], None)
                self.experiment_done = True
            elif trial == "IDLE":
                # reset timeout
                msg["idle_start"] = time.time()
                self.add_message(msg)
            else:
                with trial.lock:
                    trial.start = time.time()
                    trial.status = Trial.SCHEDULED
                    self.server.reservations.assign_trial(
                        msg["partition_id"], trial.trial_id
                    )
                    self.add_trial(trial)
        else:
            self.add_message(msg)

    def _register_msg_callback(self, msg: dict) -> None:
        """Register message callback.

        Assigns trials on worker registration if available.

        :param msg: The blacklist message from the message queue.
        """
        trial = self.controller_get_next()
        if trial is None:
            self.server.reservations.assign_trial(msg["partition_id"], None)
            self.experiment_done = True
        elif trial == "IDLE":
            # reset timeout
            msg["idle_start"] = time.time()
            self.add_message(msg)
        else:
            with trial.lock:
                trial.start = time.time()
                trial.status = Trial.SCHEDULED
                self.server.reservations.assign_trial(
                    msg["partition_id"], trial.trial_id
                )
                self.add_trial(trial)

    @staticmethod
    def _init_searchspace(searchspace: Searchspace) -> Searchspace:
        """Checks for a valid searchspace config.

        :param searchspace: The searchspace initialization argument.

        :raises ValueError: If the searchspace is invalid.

        :returns: The valid searchspace."""
        if not isinstance(searchspace, Searchspace) or searchspace is None:
            raise ValueError(
                """The experiment's search space should be an instance of
                 maggy.Searchspace, but it is {} (of type '{}').""".format(
                    str(searchspace), type(searchspace).__name__
                )
            )
        return searchspace if isinstance(searchspace, Searchspace) else Searchspace()

    @staticmethod
    def _init_controller(
        optimizer: Union[str, AbstractOptimizer], searchspace: Searchspace
    ) -> AbstractOptimizer:
        """Checks for a valid optimizer config.

        :param optimizer: The optimizer to be checked.
        :param searchspace: The searchspace for hyperparameter optimization.

        :raises KeyError: If the optimizer is given as a string, but is not in
            the dict of supported optimizers.
        :raises TypeError: If the searchspace and optimizer mismatch or the
            optimizer is of wrong type.

        :returns: The validated Optimizer
        """
        optimizer = (
            "none" if optimizer is None else optimizer
        )  # Convert None key to usable string.
        if optimizer == "none" and not searchspace.names():
            optimizer = "faulty_none"
        if isinstance(optimizer, str):
            try:
                return HyperparameterOptDriver.controller_dict[optimizer.lower()]()
            except KeyError as exc:
                raise KeyError(
                    "Unknown Optimizer. Can't initialize experiment driver."
                ) from exc
            except TypeError as exc:
                raise TypeError(
                    "Searchspace has to be empty or None to use without Optimizer."
                ) from exc
        elif isinstance(optimizer, AbstractOptimizer):
            print("Custom Optimizer initialized.")
            return optimizer
        else:
            raise TypeError(
                "The experiment's optimizer should either be an string indicating the name "
                "of an implemented optimizer (such as 'randomsearch') or an instance of "
                "maggy.optimizer.AbstractOptimizer, "
                "but it is {0} (of type '{1}').".format(
                    str(optimizer), type(optimizer).__name__
                )
            )

    @staticmethod
    def _init_earlystop_check(es_policy: Union[str, AbstractEarlyStop]) -> Callable:
        """Checks for a valid early stop policy.

        :param es_policy: The early stop policy to be checked.

        :raises TypeError: In case the policy is of wrong type or not in the
            set of supported stopping policies.

        :returns: The validated early stopping policy.
        """
        if not isinstance(es_policy, (str, AbstractEarlyStop)):
            raise TypeError(
                """The experiment's early stopping policy should either be a string
                ('median' or 'none') or a custom policy that is an instance of
                maggy.earlystop.AbstractEarlyStop, but it is {} (of type '{}').""".format(
                    str(es_policy), type(es_policy).__name__
                )
            )
        if isinstance(es_policy, str):
            if es_policy.lower() not in ["median", "none"]:
                raise TypeError(
                    """The experiment's early stopping policy should either be a
                    string ('median' or 'none') or a custom policy that is an
                    instance of maggy.earlystop.AbstractEarlyStop, but it is {}
                    (of type '{}').""".format(
                        str(es_policy), type(es_policy).__name__
                    )
                )
            rule = (
                MedianStoppingRule if es_policy.lower() == "median" else NoStoppingRule
            )
            return rule.earlystop_check
        print("Custom Early Stopping policy initialized.")
        return es_policy.earlystop_check
