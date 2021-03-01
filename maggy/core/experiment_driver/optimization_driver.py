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

from maggy import util
from maggy.searchspace import Searchspace
from maggy.optimizer import AbstractOptimizer, RandomSearch, Asha, SingleRun, GridSearch
from maggy.earlystop import AbstractEarlyStop, MedianStoppingRule, NoStoppingRule
from maggy.optimizer import bayes
from maggy.trial import Trial
from maggy.core.experiment_driver.driver import Driver
from maggy.experiment_config import AblationConfig
from maggy.core.environment.singleton import EnvSing


class OptimizationDriver(Driver):
    controller_dict = {
        "randomsearch": RandomSearch,
        "asha": Asha,
        "TPE": bayes.TPE,
        "gp": bayes.GP,
        "none": SingleRun,
        "faulty_none": None,
        "gridsearch": GridSearch,
    }

    def __init__(self, config, num_executors, log_dir):
        super().__init__(config, num_executors, log_dir)
        self._final_store = []
        self._trial_store = {}
        self.experiment_done = False
        self.maggy_log = ""

        if isinstance(
            config, AblationConfig
        ):  # Interrupt init, dirty fix for deviating config.
            return
        self.num_trials = config.num_trials
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
            raise Exception(
                "The experiment's direction should be an string (either 'min' or 'max') "
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

    def _register_callbacks(self):
        for key, call in (
            ("METRIC", self._metric_callback),
            ("BLACK", self._blacklist_callback),
            ("FINAL", self._final_callback),
            ("IDLE", self._idle_callback),
            ("REG", self._register_callback),
        ):
            self.message_callbacks[key] = call

    def controller_get_next(self, trial=None):
        return self.controller.get_suggestion(trial)

    def get_trial(self, trial_id):
        return self._trial_store[trial_id]

    def add_trial(self, trial):
        self._trial_store[trial.trial_id] = trial

    def finalize(self, job_end):
        self.job_end = job_end

        self.duration = util.seconds_to_milliseconds(self.job_end - self.job_start)

        self.duration_str = util.time_diff(self.job_start, self.job_end)

        results = self.prep_results()

        print(results)
        self.log(results)

        EnvSing.get_instance().dump(
            json.dumps(self.result, default=util.json_default_numpy),
            self.log_dir + "/result.json",
        )
        sc = util.find_spark().sparkContext
        EnvSing.get_instance().dump(self.json(sc), self.log_dir + "/maggy.json")

        return self.result

    def prep_results(self):
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
            "Total job time " + self.duration_str + "\n"
        )
        return results

    def config_to_dict(self):
        return self.searchspace.to_dict()

    def json(self, sc):
        """Get all relevant experiment information in JSON format.
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
            "app_id": str(sc.applicationId),
            "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.job_start)),
            "memory_per_executor": str(sc._conf.get("spark.executor.memory")),
            "gpus_per_executor": str(sc._conf.get("spark.executor.gpus")),
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

    def _update_result(self, trial):
        """Given a finalized trial updates the current result's best and
        worst trial.
        """

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

    def _update_maggy_log(self):
        """Creates the status of a maggy experiment with a progress bar.
        """
        return self.log_string()

    def log_string(self):
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

    def _metric_callback(self, msg):
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

    def _blacklist_callback(self, msg):
        trial = self.get_trial(msg["trial_id"])
        with trial.lock:
            trial.status = Trial.SCHEDULED
            self.server.reservations.assign_trial(msg["partition_id"], msg["trial_id"])

    def _final_callback(self, msg):
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
            trial.to_json(), self.log_dir + "/" + trial.trial_id + "/trial.json",
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

    def _idle_callback(self, msg):
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

    def _register_callback(self, msg):
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
    def _init_searchspace(searchspace):
        assert isinstance(searchspace, Searchspace) or searchspace is None, (
            "The experiment's search space should be an instance of maggy.Searchspace, but it is "
            "{0} (of type '{1}').".format(str(searchspace), type(searchspace).__name__)
        )
        return searchspace if isinstance(searchspace, Searchspace) else Searchspace()

    @staticmethod
    def _init_controller(optimizer, searchspace):
        optimizer = (
            "none" if optimizer is None else optimizer
        )  # Convert None key to usable string.
        if optimizer == "none" and not searchspace.names():
            optimizer = "faulty_none"
        if isinstance(optimizer, str):
            try:
                return OptimizationDriver.controller_dict[optimizer.lower()]()
            except KeyError as exc:
                raise Exception(
                    "Unknown Optimizer. Can't initialize experiment driver."
                ) from exc
            except TypeError as exc:
                raise Exception(
                    "Searchspace has to be empty or None to use without Optimizer."
                ) from exc
        elif isinstance(optimizer, AbstractOptimizer):
            print("Custom Optimizer initialized.")
            return optimizer
        else:
            raise Exception(
                "The experiment's optimizer should either be an string indicating the name "
                "of an implemented optimizer (such as 'randomsearch') or an instance of "
                "maggy.optimizer.AbstractOptimizer, "
                "but it is {0} (of type '{1}').".format(
                    str(optimizer), type(optimizer).__name__
                )
            )

    @staticmethod
    def _init_earlystop_check(es_policy):
        assert isinstance(
            es_policy, (str, AbstractEarlyStop)
        ), "The experiment's early stopping policy should either be a string ('median' or 'none') \
            or a custom policy that is an instance of maggy.earlystop.AbstractEarlyStop, but it is \
            {0} (of type '{1}').".format(
            str(es_policy), type(es_policy).__name__
        )
        if isinstance(es_policy, str):
            assert es_policy.lower() in [
                "median",
                "none",
            ], "The experiment's early stopping policy\
                should either be a string ('median' or 'none') or a custom policy that is an \
                instance of maggy.earlystop.AbstractEarlyStop, but it is {0} \
                (of type '{1}').".format(
                str(es_policy), type(es_policy).__name__
            )
            rule = (
                MedianStoppingRule if es_policy.lower() == "median" else NoStoppingRule
            )
            return rule.earlystop_check
        print("Custom Early Stopping policy initialized.")
        return es_policy.earlystop_check
