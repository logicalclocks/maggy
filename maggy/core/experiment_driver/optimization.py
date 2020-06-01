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

from maggy import util
from maggy.searchspace import Searchspace
from maggy.optimizer import AbstractOptimizer, RandomSearch, Asha, SingleRun
from maggy.earlystop import AbstractEarlyStop, MedianStoppingRule, NoStoppingRule
from maggy.core.experiment_driver import base


class Driver(base.Driver):
    def __init__(
        self,
        num_trials,
        optimizer,
        searchspace,
        direction,
        es_policy,
        es_interval,
        es_min,
        name,
        description,
        num_executors,
        hb_interval,
        log_dir,
    ):
        # num_trials default 1
        # direction default 'max'
        super().__init__(
            name, description, direction, num_executors, hb_interval, log_dir
        )

        # CONTEXT-SPECIFIC EXPERIMENT SETUP
        self.num_trials = num_trials

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

        if optimizer is None:
            if len(self.searchspace.names()) == 0:
                self.controller = SingleRun()
            else:
                raise Exception(
                    "Searchspace has to be empty or None to use without optimizer"
                )
        elif isinstance(optimizer, str):
            if optimizer.lower() == "randomsearch":
                self.controller = RandomSearch()
            elif optimizer.lower() == "asha":
                self.controller = Asha()
            elif optimizer.lower() == "none":
                if len(self.searchspace.names()) == 0:
                    self.controller = SingleRun()
                else:
                    raise Exception(
                        "Searchspace has to be empty or None to use without Optimizer."
                    )
            else:
                raise Exception(
                    "Unknown Optimizer. Can't initialize experiment driver."
                )
        elif isinstance(optimizer, AbstractOptimizer):
            self.controller = optimizer
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

        self.es_interval = es_interval
        self.es_min = es_min

        self.result = {"best_val": "n.a.", "num_trials": 0, "early_stopped": 0}

        # Init controller and set references to data
        self.controller.num_trials = self.num_trials
        self.controller.searchspace = self.searchspace
        self.controller.trial_store = self._trial_store
        self.controller.final_store = self._final_store
        self.controller.direction = self.direction
        self.controller.initialize()

    def controller_get_next(self, trial=None):
        return self.controller.get_suggestion(trial)

    def prep_results(self):
        _ = self.controller.finalize_experiment(self._final_store)
        results = (
            "\n------ "
            + self.controller.name()
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
        return results

    def config_to_dict(self):
        return self.searchspace.to_dict()

    def log_string(self):
        log = (
            "Maggy Optimization "
            + str(self.result["num_trials"])
            + "/"
            + str(self.num_trials)
            + " ("
            + str(self.result["early_stopped"])
            + ") "
            + util._progress_bar(self.result["num_trials"], self.num_trials)
            + " - BEST "
            + json.dumps(self.result["best_hp"])
            + " - metric "
            + str(self.result["best_val"])
        )
        return log
