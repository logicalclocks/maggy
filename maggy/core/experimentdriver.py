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
The experiment driver implements the functionality for scheduling trials on
maggy.
"""
import queue
import threading
import json
import os
import secrets
import time
from datetime import datetime

from hops import constants as hopsconstants
from hops import hdfs as hopshdfs
from hops import util as hopsutil
from hops.experiment_impl.util import experiment_utils

from maggy import util
from maggy.optimizer import AbstractOptimizer, RandomSearch, Asha, SingleRun
from maggy.core import rpc
from maggy.trial import Trial
from maggy.earlystop import AbstractEarlyStop, MedianStoppingRule, NoStoppingRule
from maggy.searchspace import Searchspace

from maggy.ablation.ablator import AbstractAblator
from maggy.ablation.ablator.loco import LOCO
from maggy.ablation.ablationstudy import AblationStudy

driver_secret = None


class ExperimentDriver(object):

    SECRET_BYTES = 8

    # @Moritz:
    # for now, we infer the experiment type (an optimization experiment or an ablation study)
    # using keyword arguments, and set self.experiment_type to an according string.
    # Some of these arguments are required for any maggy experiment:
    # num_trials, name, num_executors, hb_interval, description, app_dir, log_dir, trial_dir
    # while some are specific to the type of experiment. For example, if the ExperimentDriver constructor
    # is called with a `searchspace` parameter, we infer that it is a hyperparameter optimization task,
    # and if it is called with `ablationstudy` parameter, we infer that it's an ablation study.

    def __init__(self, experiment_type, **kwargs):

        global driver_secret

        # COMMON EXPERIMENT SETUP
        self._final_store = []
        self._trial_store = {}
        self.num_executors = kwargs.get("num_executors")
        self._message_q = queue.Queue()
        self.name = kwargs.get("name")
        self.experiment_done = False
        self.worker_done = False
        self.hb_interval = kwargs.get("hb_interval")
        self.description = kwargs.get("description")
        self.experiment_type = experiment_type
        self.es_interval = kwargs.get("es_interval")
        self.es_min = kwargs.get("es_min")

        # TYPE-SPECIFIC EXPERIMENT SETUP
        if self.experiment_type == "optimization":
            # set up an optimization experiment

            self.num_trials = kwargs.get("num_trials", 1)

            searchspace = kwargs.get("searchspace")
            if isinstance(searchspace, Searchspace):
                self.searchspace = searchspace
            elif searchspace is None:
                self.searchspace = Searchspace()
            else:
                raise Exception(
                    "The experiment's search space should be an instance of maggy.Searchspace, "
                    "but it is {0} (of type '{1}').".format(
                        str(searchspace), type(searchspace).__name__
                    )
                )

            optimizer = kwargs.get("optimizer")

            if optimizer is None:
                if len(self.searchspace.names()) == 0:
                    self.optimizer = SingleRun()
                else:
                    raise Exception(
                        "Searchspace has to be empty or None to use without optimizer"
                    )
            elif isinstance(optimizer, str):
                if optimizer.lower() == "randomsearch":
                    self.optimizer = RandomSearch()
                elif optimizer.lower() == "asha":
                    self.optimizer = Asha()
                elif optimizer.lower() == "none":
                    if len(self.searchspace.names()) == 0:
                        self.optimizer = SingleRun()
                    else:
                        raise Exception(
                            "Searchspace has to be empty or None to use without Optimizer."
                        )
                else:
                    raise Exception(
                        "Unknown Optimizer. Can't initialize experiment driver."
                    )
            elif isinstance(optimizer, AbstractOptimizer):
                self.optimizer = optimizer
                print("Custom Optimizer initialized.")
            else:
                raise Exception(
                    "The experiment's optimizer should either be an string indicating the name "
                    "of an implemented optimizer (such as 'randomsearch') or an instance of "
                    "maggy.optimizer.AbstractOptimizer, "
                    "but it is {0} (of type '{1}').".format(
                        str(optimizer), type(optimizer).__name__
                    )
                )

            direction = kwargs.get("direction", "max")
            if isinstance(direction, str) and direction.lower() in ["min", "max"]:
                self.direction = direction.lower()
            else:
                raise Exception(
                    "The experiment's direction should be an string (either 'min' or 'max') "
                    "but it is {0} (of type '{1}').".format(
                        str(direction), type(direction).__name__
                    )
                )

            es_policy = kwargs.get("es_policy")
            if isinstance(es_policy, str):
                if es_policy.lower() == "median":
                    self.earlystop_check = MedianStoppingRule.earlystop_check
                elif es_policy.lower() == "none":
                    self.earlystop_check = NoStoppingRule.earlystop_check
                else:
                    raise Exception(
                        "The experiment's early stopping policy should either be a string ('median' or 'none') "
                        "or a custom policy that is an instance of maggy.earlystop.AbstractEarlyStop, "
                        "but it is {0} (of type '{1}').".format(
                            str(es_policy), type(es_policy).__name__
                        )
                    )
            elif isinstance(es_policy, AbstractEarlyStop):
                self.earlystop_check = es_policy.earlystop_check
                print("Custom Early Stopping policy initialized.")
            else:
                raise Exception(
                    "The experiment's early stopping policy should either be a string ('median' or 'none') "
                    "or a custom policy that is an instance of maggy.earlystop.AbstractEarlyStop, "
                    "but it is {0} (of type '{1}').".format(
                        str(es_policy), type(es_policy).__name__
                    )
                )

            self.es_interval = kwargs.get("es_interval")
            self.es_min = kwargs.get("es_min")

            self.result = {"best_val": "n.a.", "num_trials": 0, "early_stopped": 0}

        elif self.experiment_type == "ablation":
            # set up an ablation study experiment
            self.earlystop_check = NoStoppingRule.earlystop_check

            ablation_study = kwargs.get("ablation_study")
            if isinstance(ablation_study, AblationStudy):
                self.ablation_study = ablation_study
            else:
                raise Exception(
                    "The experiment's ablation study configuration should be an instance of "
                    "maggy.ablation.AblationStudy, "
                    "but it is {0} (of type '{1}').".format(
                        str(ablation_study), type(ablation_study).__name__
                    )
                )

            searchspace = kwargs.get("searchspace")
            if not searchspace:
                self.searchspace = Searchspace()
            else:
                raise Exception(
                    "The experiment's search space should be None for ablation experiments, "
                    "but it is {0} (of type '{1}').".format(
                        str(searchspace), type(searchspace).__name__
                    )
                )

            ablator = kwargs.get("ablator")
            if isinstance(ablator, str):
                if ablator.lower() == "loco":
                    self.ablator = LOCO(ablation_study, self._final_store)
                    self.num_trials = self.ablator.get_number_of_trials()
                    if self.num_executors > self.num_trials:
                        self.num_executors = self.num_trials
                else:
                    raise Exception(
                        "The experiment's ablation study policy should either be a string ('loco') "
                        "or a custom policy that is an instance of maggy.ablation.ablation.AbstractAblator, "
                        "but it is {0} (of type '{1}').".format(
                            str(ablator), type(ablator).__name__
                        )
                    )
            elif isinstance(ablator, AbstractAblator):
                self.ablator = ablator
                print("Custom Ablator initialized. \n")
            else:
                raise Exception(
                    "The experiment's ablation study policy should either be a string ('loco') "
                    "or a custom policy that is an instance of maggy.ablation.ablation.AbstractAblator, "
                    "but it is {0} (of type '{1}').".format(
                        str(ablator), type(ablator).__name__
                    )
                )

            self.result = {"best_val": "n.a.", "num_trials": 0, "early_stopped": "n.a"}
        else:
            raise Exception(
                "Unknown experiment type. experiment_type should be either 'optimization' or 'ablation', "
                "but it is {0}.".format(str(self.experiment_type))
            )

        # FINALIZE EXPERIMENT SETUP
        self.server = rpc.Server(self.num_executors)
        if not driver_secret:
            driver_secret = self._generate_secret(ExperimentDriver.SECRET_BYTES)
        self._secret = driver_secret
        self.job_start = datetime.now()
        self.executor_logs = ""
        self.maggy_log = ""
        self.log_lock = threading.RLock()
        self.log_file = kwargs.get("log_dir") + "/maggy.log"
        self.log_dir = kwargs.get("log_dir")
        self.exception = None

        # Open File desc for HDFS to log
        if not hopshdfs.exists(self.log_file):
            hopshdfs.dump("", self.log_file)
        self.fd = hopshdfs.open_file(self.log_file, flags="w")

    def init(self, job_start):

        self.server_addr = self.server.start(self)

        self.job_start = job_start

        if self.experiment_type == "optimization":
            # Set references to data in optimizer
            self.optimizer.num_trials = self.num_trials
            self.optimizer.searchspace = self.searchspace
            self.optimizer.trial_store = self._trial_store
            self.optimizer.final_store = self._final_store
            self.optimizer.direction = self.direction
            self.optimizer.initialize()
        elif self.experiment_type == "ablation":
            # set references to data in ablator
            self.ablator.ablation_study = self.ablation_study
            self.ablator.final_store = self._final_store
            self.ablator.initialize()

        self._start_worker()

    def finalize(self, job_end):

        results = ""

        if self.experiment_type == "optimization":

            _ = self.optimizer.finalize_experiment(self._final_store)

            self.job_end = job_end

            self.duration = experiment_utils._seconds_to_milliseconds(
                self.job_end - self.job_start
            )

            self.duration_str = experiment_utils._time_diff(
                self.job_start, self.job_end
            )

            results = (
                "\n------ "
                + self.optimizer.name()
                + " Results ------ direction("
                + self.direction
                + ") \n"
                "BEST combination "
                + json.dumps(self.result["best_hp"])
                + " -- metric "
                + str(self.result["best_val"])
                + "\n"
                "WORST combination "
                + json.dumps(self.result["worst_hp"])
                + " -- metric "
                + str(self.result["worst_val"])
                + "\n"
                "AVERAGE metric -- " + str(self.result["avg"]) + "\n"
                "EARLY STOPPED Trials -- " + str(self.result["early_stopped"]) + "\n"
                "Total job time " + self.duration_str + "\n"
            )

        elif self.experiment_type == "ablation":

            _ = self.ablator.finalize_experiment(self._final_store)
            self.job_end = job_end

            self.duration = experiment_utils._seconds_to_milliseconds(
                self.job_end - self.job_start
            )

            self.duration_str = experiment_utils._time_diff(
                self.job_start, self.job_end
            )

            results = (
                "\n------ "
                + self.ablator.name()
                + " Results ------ \n"
                + "BEST Config Excludes "
                + json.dumps(self.result["best_config"])
                + " -- metric "
                + str(self.result["best_val"])
                + "\n"
                + "WORST Config Excludes "
                + json.dumps(self.result["worst_config"])
                + " -- metric "
                + str(self.result["worst_val"])
                + "\n"
                + "AVERAGE metric -- "
                + str(self.result["avg"])
                + "\n"
                + "Total Job Time "
                + self.duration_str
                + "\n"
            )

        print(results)

        self._log(results)

        hopshdfs.dump(
            json.dumps(self.result, default=util.json_default_numpy),
            self.log_dir + "/result.json",
        )
        sc = hopsutil._find_spark().sparkContext
        hopshdfs.dump(self.json(sc), self.log_dir + "/maggy.json")

        return self.result

    def get_trial(self, trial_id):
        return self._trial_store[trial_id]

    def add_trial(self, trial):
        self._trial_store[trial.trial_id] = trial

    def add_message(self, msg):
        self._message_q.put(msg)

    def _start_worker(self):
        def _target_function(self):

            try:
                while not self.worker_done:
                    trial = None
                    # get a message
                    try:
                        msg = self._message_q.get_nowait()
                    except queue.Empty:
                        msg = {"type": None}

                    # depending on message do the work
                    # 1. METRIC
                    if msg["type"] == "METRIC":
                        # append executor logs if in the message
                        logs = msg.get("logs", None)
                        if logs is not None:
                            with self.log_lock:
                                self.executor_logs = self.executor_logs + logs

                        step = None
                        if msg["trial_id"] is not None and msg["data"] is not None:
                            step = self.get_trial(msg["trial_id"]).append_metric(
                                msg["data"]
                            )

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
                                            self._log(e)
                                            to_stop = None
                                        if to_stop is not None:
                                            self._log(
                                                "Trials to stop: {}".format(to_stop)
                                            )
                                            self.get_trial(to_stop).set_early_stop()

                        # 2. BLACKLIST the trial
                    elif msg["type"] == "BLACK":
                        trial = self.get_trial(msg["trial_id"])
                        with trial.lock:
                            trial.status = Trial.SCHEDULED
                            self.server.reservations.assign_trial(
                                msg["partition_id"], msg["trial_id"]
                            )

                    # 3. FINAL
                    elif msg["type"] == "FINAL":
                        # set status
                        # get trial only once
                        trial = self.get_trial(msg["trial_id"])

                        logs = msg.get("logs", None)
                        if logs is not None:
                            with self.log_lock:
                                self.executor_logs = self.executor_logs + logs

                        # finalize the trial object
                        with trial.lock:
                            trial.status = Trial.FINALIZED
                            trial.final_metric = msg["data"]
                            trial.duration = experiment_utils._seconds_to_milliseconds(
                                time.time() - trial.start
                            )

                        # move trial to the finalized ones
                        self._final_store.append(trial)
                        self._trial_store.pop(trial.trial_id)

                        # update result dictionary
                        self._update_result(trial)
                        # keep for later in case tqdm doesn't work
                        self.maggy_log = self._update_maggy_log()
                        self._log(self.maggy_log)

                        hopshdfs.dump(
                            trial.to_json(),
                            self.log_dir + "/" + trial.trial_id + "/trial.json",
                        )

                        # assign new trial
                        if self.experiment_type == "optimization":
                            trial = self.optimizer.get_suggestion(trial)
                        elif self.experiment_type == "ablation":
                            trial = self.ablator.get_trial(trial)
                        if trial is None:
                            self.server.reservations.assign_trial(
                                msg["partition_id"], None
                            )
                            self.experiment_done = True
                        elif trial == "IDLE":
                            self.add_message(
                                {
                                    "type": "IDLE",
                                    "partition_id": msg["partition_id"],
                                    "idle_start": time.time(),
                                }
                            )
                            self.server.reservations.assign_trial(
                                msg["partition_id"], None
                            )
                        else:
                            with trial.lock:
                                trial.start = time.time()
                                trial.status = Trial.SCHEDULED
                                self.server.reservations.assign_trial(
                                    msg["partition_id"], trial.trial_id
                                )
                                self.add_trial(trial)

                    # 4. Let executor be idle
                    elif msg["type"] == "IDLE":
                        # execute only every 0.1 seconds but do not block thread
                        if (
                            self.experiment_type == "optimization"
                            and time.time() - msg["idle_start"] > 0.1
                        ):
                            trial = self.optimizer.get_suggestion()
                            if trial is None:
                                self.server.reservations.assign_trial(
                                    msg["partition_id"], None
                                )
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
                        elif self.experiment_type == "optimization":
                            self.add_message(msg)

                    # 4. REG
                    elif msg["type"] == "REG":
                        if self.experiment_type == "optimization":
                            trial = self.optimizer.get_suggestion()
                        elif self.experiment_type == "ablation":
                            trial = self.ablator.get_trial()
                        if trial is None:
                            self.server.reservations.assign_trial(
                                msg["partition_id"], None
                            )
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
            except Exception as exc:
                # Exception can't be propagated to parent thread
                # therefore log the exception and fail experiment
                self._log(exc)
                self.exception = exc
                self.server.stop()

        t = threading.Thread(target=_target_function, args=(self,))
        t.daemon = True
        t.start()

    def stop(self):
        """Stop the Driver's worker thread and server."""
        self.worker_done = True
        self.server.stop()
        self.fd.flush()
        self.fd.close()

    def json(self, sc):
        """Get all relevant experiment information in JSON format.
        """
        user = None
        if hopsconstants.ENV_VARIABLES.HOPSWORKS_USER_ENV_VAR in os.environ:
            user = os.environ[hopsconstants.ENV_VARIABLES.HOPSWORKS_USER_ENV_VAR]

        experiment_json = {
            "project": hopshdfs.project_name(),
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
            "experiment_type": self.experiment_type,
        }

        if self.experiment_type == "optimization":
            experiment_json["hyperparameter_space"] = json.dumps(
                self.searchspace.to_dict()
            )
            experiment_json["function"] = self.optimizer.name()
        elif self.experiment_type == "ablation":
            experiment_json["ablation_study"] = json.dumps(
                self.ablation_study.to_dict()
            )
            experiment_json["ablator"] = self.ablator.name()

        if self.experiment_done:
            experiment_json["status"] = "FINISHED"
            experiment_json["finished"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self.job_end)
            )
            experiment_json["duration"] = self.duration
            if self.experiment_type == "optimization":
                experiment_json["hyperparameter"] = json.dumps(self.result["best_hp"])
            experiment_json["metric"] = self.result["best_val"]

        else:
            experiment_json["status"] = "RUNNING"

        return json.dumps(experiment_json, default=util.json_default_numpy)

    def _generate_secret(self, nbytes):
        """Generates a secret to be used by all clients during the experiment
        to authenticate their messages with the experiment driver.
        """
        return secrets.token_hex(nbytes=nbytes)

    def _update_result(self, trial):
        """Given a finalized trial updates the current result's best and
        worst trial.
        """

        metric = trial.final_metric
        param_string = trial.params
        trial_id = trial.trial_id

        if self.experiment_type == "optimization":
            # First finalized trial
            if self.result.get("best_id", None) is None:
                self.result = {
                    "best_id": trial_id,
                    "best_val": metric,
                    "best_hp": param_string,
                    "worst_id": trial_id,
                    "worst_val": metric,
                    "worst_hp": param_string,
                    "avg": metric,
                    "metric_list": [metric],
                    "num_trials": 1,
                    "early_stopped": 0,
                }

                if trial.early_stop:
                    self.result["early_stopped"] += 1

                return
            if self.direction == "max":
                if metric > self.result["best_val"]:
                    self.result["best_val"] = metric
                    self.result["best_id"] = trial_id
                    self.result["best_hp"] = param_string
                if metric < self.result["worst_val"]:
                    self.result["worst_val"] = metric
                    self.result["worst_id"] = trial_id
                    self.result["worst_hp"] = param_string
            elif self.direction == "min":
                if metric < self.result["best_val"]:
                    self.result["best_val"] = metric
                    self.result["best_id"] = trial_id
                    self.result["best_hp"] = param_string
                if metric > self.result["worst_val"]:
                    self.result["worst_val"] = metric
                    self.result["worst_id"] = trial_id
                    self.result["worst_hp"] = param_string

        elif self.experiment_type == "ablation":

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
                }
                return

            # for ablation we always consider 'max' as "the direction"
            if metric > self.result["best_val"]:
                self.result["best_id"] = trial_id
                self.result["best_val"] = metric
                self.result["best_config"] = param_string
            elif metric < self.result["worst_val"]:
                self.result["worst_id"] = trial_id
                self.result["worst_val"] = metric
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
        finished = self.result["num_trials"]
        log = ""

        if self.experiment_type == "optimization":
            log = (
                "Maggy Optimization "
                + str(finished)
                + "/"
                + str(self.num_trials)
                + " ("
                + str(self.result["early_stopped"])
                + ") "
                + util._progress_bar(finished, self.num_trials)
                + " - BEST "
                + json.dumps(self.result["best_hp"])
                + " - metric "
                + str(self.result["best_val"])
            )

        elif self.experiment_type == "ablation":
            log = (
                "Maggy Ablation "
                + str(finished)
                + "/"
                + str(self.num_trials)
                + util._progress_bar(finished, self.num_trials)
                + " - BEST Excludes"
                + json.dumps(self.result["best_config"])
                + " - metric "
                + str(self.result["best_val"])
            )

        return log

    def _get_logs(self):
        """Return current experiment status and executor logs to send them to
        spark magic.
        """
        with self.log_lock:
            temp = self.executor_logs
            # clear the executor logs since they are being sent
            self.executor_logs = ""
            return self.result, temp

    def _log(self, log_msg):
        """Logs a string to the maggy driver log file.
        """
        msg = datetime.now().isoformat() + ": " + str(log_msg)
        self.fd.write((msg + "\n").encode())
