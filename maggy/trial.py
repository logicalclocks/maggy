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

import json
import threading
import hashlib

from maggy import util


class Trial(object):
    """A Trial object contains all relevant information about the evaluation
    of an hyperparameter combination.

    It is used as shared memory between
    the worker thread and rpc server thread. The server thread performs only
    lookups on the `early_stop` and `params` attributes.
    """

    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    FINALIZED = "FINALIZED"

    def __init__(self, params, trial_type="optimization"):
        """Create a new trial object from a hyperparameter combination
        ``params``.

        :param params: A dictionary of Hyperparameters as key value pairs.
        :type params: dict
        """
        # XXX before merge, we should remove the default value for trial_type
        # and make sure everywhere Trial() is called (e.g. in all optimizers)
        # trial_type is passed
        # @Moritz

        self.trial_type = trial_type
        # XXX temp fix, have to come up with abstractions
        if self.trial_type == "optimization":
            self.trial_id = Trial._generate_id(params)
        elif self.trial_type == "ablation":
            serializable_params = {
                "ablated_feature": params.get("ablated_feature", None),
                "ablated_layer": params.get("ablated_layer", None),
            }
            self.trial_id = Trial._generate_id(serializable_params)
        self.params = params
        self.status = Trial.PENDING
        self.early_stop = False
        self.final_metric = None
        self.metric_history = []
        self.step_history = []
        self.metric_dict = {}
        self.start = None
        self.duration = None
        self.lock = threading.RLock()

    def get_early_stop(self):
        """Return the early stopping flag of the trial."""
        with self.lock:
            return self.early_stop

    def set_early_stop(self):
        """Set the early stopping flag of the trial to true."""
        with self.lock:
            self.early_stop = True

    def append_metric(self, metric_data):
        """Append a metric from the heartbeats to the history."""
        with self.lock:
            # from python 3.7 dicts are insertion ordered,
            # so two of these data structures can be removed
            if (
                metric_data["step"] not in self.metric_dict
                and metric_data["value"] is not None
            ):
                self.metric_dict[metric_data["step"]] = metric_data["value"]
                self.metric_history.append(metric_data["value"])
                self.step_history.append(metric_data["step"])
                # return step number to indicate that it was a new unique step
                return metric_data["step"]
            # return None to indicate that no new step has finished
            return None

    @classmethod
    def _generate_id(cls, params):
        """
        Class method to generate a hash from a hyperparameter dictionary.

        All keys in the dictionary have to be strings. The hash is a to 16
        characters truncated md5 hash and stable across processes.

        :param params: Hyperparameters
        :type params: dictionary
        :raises ValueError: All hyperparameter names have to be strings.
        :raises ValueError: Hyperparameters need to be a dictionary.
        :return: Sixteen character truncated md5 hash
        :rtype: str
        """

        # ensure params is a dictionary
        if isinstance(params, dict):
            # check that all keys are strings
            if False in set(isinstance(k, str) for k in params.keys()):
                raise ValueError("All hyperparameter names have to be strings.")

            return hashlib.md5(
                json.dumps(params, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]

        raise ValueError("Hyperparameters need to be a dictionary.")

    def to_json(self):
        return json.dumps(self.to_dict(), default=util.json_default_numpy)

    def to_dict(self):
        obj_dict = {"__class__": self.__class__.__name__}

        temp_dict = self.__dict__.copy()
        temp_dict.pop("lock")
        temp_dict.pop("start")

        obj_dict.update(temp_dict)

        return obj_dict

    @classmethod
    def from_json(cls, json_str):
        """Creates a Trial instance from a previously json serialized Trial
        object instance.

        :param json_str: String containing the object.
        :type json_str: str
        :raises ValueError: json_str is not a Trial object.
        :return: Instantiated object instance of Trial.
        :rtype: Trial
        """

        temp_dict = json.loads(json_str)
        if temp_dict.get("__class__", None) != "Trial":
            raise ValueError("json_str is not a Trial object.")
        if temp_dict.get("params", None) is not None:
            instance = cls(temp_dict.get("params"))
            instance.trial_id = temp_dict["trial_id"]
            instance.status = temp_dict["status"]
            instance.early_stop = temp_dict.get("early_stop", False)
            instance.final_metric = temp_dict["final_metric"]
            instance.metric_history = temp_dict["metric_history"]
            instance.duration = temp_dict["duration"]

        return instance
