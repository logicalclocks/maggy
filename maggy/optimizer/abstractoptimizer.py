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
from abc import abstractmethod

from maggy.core import controller


class AbstractOptimizer(controller.Controller):
    def __init__(self):
        self.searchspace = None
        self.num_trials = None
        self.trial_store = None
        self.final_store = None
        self.direction = None

    @abstractmethod
    def initialize(self):
        """
        A hook for the developer to initialize the optimizer.
        """
        pass

    @abstractmethod
    def get_next_trial(self, trial=None):
        """
        Return a `Trial` to be assigned to an executor, or `None` if there are
        no trials remaining in the experiment.

        :rtype: `Trial` or `None`
        """
        pass

    def finalize(self, result, trials):
        """
        This method will be called before finishing the experiment. Developers
        can override or extend this method e.g. for cleanup or extra logging.
        Maggy expects two values to be returned, a dictionary and a string
        to be printed. The dictionary will be merged with the `result` dict of
        Maggy, persisted and returned to the user. The finalized `trials` can
        be used to compute additional statistics to return.
        The returned string representation of the result should be human readable
        and will be printed and written to the logs.

        :param result: Results of the experiment as dictionary.
        :type result: dict
        :param trials: The finalized trial objects as a list.
        :type trials: list
        :return: result metrics, result string representation
        :rtype: dict, str
        """
        result_dict = {}
        result_str = (
            "\n------ "
            + self.name()
            + " Results ------ direction("
            + self.direction
            + ") \n"
            "BEST configuration "
            + json.dumps(result["best_config"])
            + " -- metric "
            + str(result["best_val"])
            + "\n"
            "WORST combination "
            + json.dumps(result["worst_config"])
            + " -- metric "
            + str(result["worst_val"])
            + "\n"
            "AVERAGE metric -- " + str(result["avg"]) + "\n"
            "EARLY STOPPED Trials -- " + str(result["early_stopped"]) + "\n"
            "Total job time " + result["duration_str"] + "\n"
        )
        return result_dict, result_str
