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
from abc import ABC, abstractmethod
from datetime import datetime

from hops import constants as hopsconstants
from hops import hdfs as hopshdfs
from hops import util as hopsutil
from hops.experiment_impl.util import experiment_utils

from maggy import util
from maggy.core import rpc
from maggy.trial import Trial
from maggy.earlystop import NoStoppingRule

driver_secret = None


class Driver(ABC):

    SECRET_BYTES = 8

    def __init__(
        self, name, description, direction, num_executors, hb_interval, log_dir
    ):
        global driver_secret

        # COMMON EXPERIMENT SETUP
        # Functionality inits
        self._final_store = []
        self._trial_store = {}
        self.num_executors = num_executors
        self._message_q = queue.Queue()
        self.experiment_done = False
        self.worker_done = False
        self.hb_interval = hb_interval
        self.server = rpc.Server(self.num_executors)

        if not driver_secret:
            driver_secret = self._generate_secret(self.SECRET_BYTES)

        self._secret = driver_secret
        self.executor_logs = ""
        self.maggy_log = ""
        self.log_lock = threading.RLock()
        self.log_file = log_dir + "/maggy.log"
        self.log_dir = log_dir
        self.exception = None

        if isinstance(direction, str) and direction.lower() in ["min", "max"]:
            self.direction = direction.lower()
        else:
            raise Exception(
                "The experiment's direction should be an string (either 'min' or 'max') "
                "but it is {0} (of type '{1}').".format(
                    str(direction), type(direction).__name__
                )
            )

        # Open File desc for HDFS to log
        if not hopshdfs.exists(self.log_file):
            hopshdfs.dump("", self.log_file)
        self.fd = hopshdfs.open_file(self.log_file, flags="w")

        # overwritten for optimization
        self.es_interval = None
        self.es_min = None

        # Metadata
        self.name = name
        self.description = description

    def init(self, job_start):
        self.server_addr = self.server.start(self)
        self.job_start = job_start
        self._start_worker()

    def finalize(self, job_end):
        self.job_end = job_end

        self.result["duration"] = experiment_utils._seconds_to_milliseconds(
            self.job_end - self.job_start
        )

        self.result["duration_str"] = experiment_utils._time_diff(
            self.job_start, self.job_end
        )

        controller_results, result_str = self.controller.finalize(
            self.result, self._final_store
        )
        if not (isinstance(controller_results, dict) and isinstance(result_str, str)):
            raise TypeError(
                "The `finalize` method of the used controller returns a tuple "
                "with element types ({}, {}) instead of the required (dict, "
                "str) tuple.".format(type(controller_results), type(result_str))
            )
        self.result.update(controller_results)

        print(result_str)
        self._log(result_str)

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
                time_earlystop_check = (
                    time.time()
                )  # only used by earlystop-supporting experiments

                while not self.worker_done:
                    trial = None
                    # get a message
                    try:
                        msg = self._message_q.get_nowait()
                    except queue.Empty:
                        msg = {"type": None}

                    if self.earlystop_check != NoStoppingRule.earlystop_check:
                        if (time.time() - time_earlystop_check) >= self.es_interval:
                            time_earlystop_check = time.time()

                            # pass currently running trials to early stop component
                            if len(self._final_store) > self.es_min:
                                self._log("Check for early stopping.")
                                try:
                                    to_stop = self.earlystop_check(
                                        self._trial_store,
                                        self._final_store,
                                        self.direction,
                                    )
                                except Exception as e:
                                    self._log(e)
                                    to_stop = []
                                if len(to_stop) > 0:
                                    self._log("Trials to stop: {}".format(to_stop))
                                for trial_id in to_stop:
                                    self.get_trial(trial_id).set_early_stop()

                    # depending on message do the work
                    # 1. METRIC
                    if msg["type"] == "METRIC":
                        # append executor logs if in the message
                        logs = msg.get("logs", None)
                        if logs is not None:
                            with self.log_lock:
                                self.executor_logs = self.executor_logs + logs

                        if msg["trial_id"] is not None and msg["data"] is not None:
                            self.get_trial(msg["trial_id"]).append_metric(msg["data"])

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
                        trial = self.controller.get_next_trial(trial)
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
                        if time.time() - msg["idle_start"] > 0.1:
                            trial = self.controller.get_next_trial()
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
                        else:
                            self.add_message(msg)

                    # 4. REG
                    elif msg["type"] == "REG":
                        trial = self.controller.get_next_trial()
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

    @abstractmethod
    def config_to_dict(self):
        pass

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
            "experiment_type": self.controller.name(),
        }

        experiment_json["controller"] = self.controller.name()
        experiment_json["config"] = json.dumps(self.config_to_dict())

        if self.experiment_done:
            experiment_json["status"] = "FINISHED"
            experiment_json["finished"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self.job_end)
            )
            experiment_json["duration"] = self.result["duration"]
            experiment_json["config"] = json.dumps(self.result["best_config"])
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

    @abstractmethod
    def log_string(self):
        pass

    def _update_maggy_log(self):
        """Creates the status of a maggy experiment with a progress bar.
        """
        return self.log_string()

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
