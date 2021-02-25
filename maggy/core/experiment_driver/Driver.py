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
import secrets
import time
from abc import ABC, abstractmethod
from datetime import datetime

from maggy import util
from maggy.core import rpc
from maggy.trial import Trial
from maggy.earlystop import NoStoppingRule
from maggy.core.environment.singleton import EnvSing

driver_secret = None


class Driver(ABC):

    SECRET_BYTES = 8

    def __init__(
        self, name, description, direction, num_executors, hb_interval, log_dir
    ):
        global driver_secret

        # COMMON EXPERIMENT SETUP
        # Functionality inits
        self.exp_type = None
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

        env = EnvSing.get_instance()
        # Open File desc for HDFS to log
        if not env.exists(self.log_file):
            env.dump("", self.log_file)
        self.fd = env.open_file(self.log_file, flags="w")

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

    @abstractmethod
    def prep_results(self):
        pass

    def finalize(self, job_end):
        self.job_end = job_end

        self.duration = util.seconds_to_milliseconds(self.job_end - self.job_start)

        self.duration_str = util.time_diff(self.job_start, self.job_end)

        results = self.prep_results()

        print(results)
        self._log(results)

        EnvSing.get_instance().dump(
            json.dumps(self.result, default=util.json_default_numpy),
            self.log_dir + "/result.json",
        )
        sc = util.find_spark().sparkContext
        EnvSing.get_instance().dump(self.json(sc), self.log_dir + "/maggy.json")

        return self.result

    def get_trial(self, trial_id):
        return self._trial_store[trial_id]

    def add_trial(self, trial):
        self._trial_store[trial.trial_id] = trial

    def add_message(self, msg):
        self._message_q.put(msg)

    @abstractmethod
    def controller_get_next(self, trial=None):
        # TODO this won't be necessary if ablator and optimizer implement same
        # interface
        pass

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
                            trial.duration = util.seconds_to_milliseconds(
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

                        EnvSing.get_instance().dump(
                            trial.to_json(),
                            self.log_dir + "/" + trial.trial_id + "/trial.json",
                        )

                        # assign new trial
                        trial = self.controller_get_next(trial)
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
                            trial = self.controller_get_next()
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
                        trial = self.controller_get_next()
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
        user = EnvSing.get_instance().get_user()

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
        self.fd.write(EnvSing.get_instance().str_or_byte(msg + "\n"))
