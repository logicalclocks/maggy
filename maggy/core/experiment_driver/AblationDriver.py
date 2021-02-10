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
from maggy.earlystop import NoStoppingRule
from maggy.ablation.ablationstudy import AblationStudy
from maggy.ablation.ablator.loco import LOCO
from maggy.ablation.ablator import AbstractAblator
from maggy.core.experiment_driver.OptimizationDriver import OptimizationDriver


class AblationDriver(OptimizationDriver):
    def __init__(self, config, num_executors, log_dir):
        super().__init__(config, num_executors, log_dir)
        # set up an ablation study experiment
        self.earlystop_check = NoStoppingRule.earlystop_check

        if isinstance(config.ablation_study, AblationStudy):
            self.ablation_study = config.ablation_study
        else:
            raise Exception(
                "The experiment's ablation study configuration should be an instance of "
                "maggy.ablation.AblationStudy, "
                "but it is {0} (of type '{1}').".format(
                    str(config.ablation_study), type(config.ablation_study).__name__
                )
            )

        if isinstance(config.ablator, str) and config.ablator.lower() == "loco":
            self.controller = LOCO(config.ablation_study, self._final_store)
            self.num_trials = self.controller.get_number_of_trials()
            self.num_executors = min(self.num_executors, self.num_trials)
        elif isinstance(config.ablator, AbstractAblator):
            self.controller = config.ablator
            print("Custom Ablator initialized. \n")
        else:
            raise Exception(
                "The experiment's ablation study policy should either be a string ('loco') "
                "or a custom policy that is an instance of maggy.ablation.ablation.AbstractAblator,"
                " but it is {0} (of type '{1}').".format(
                    str(config.ablator), type(config.ablator).__name__
                )
            )

        self.result = {"best_val": "n.a.", "num_trials": 0, "early_stopped": "n.a"}

        # Init controller and set references to data in ablator
        self.controller.ablation_study = self.ablation_study
        self.controller.final_store = self._final_store
        self.controller.initialize()

    def controller_get_next(self, trial=None):
        return self.controller.get_trial(trial)

    def prep_results(self):
        _ = self.controller.finalize_experiment(self._final_store)
        results = (
            "\n------ "
            + self.controller.name()
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
        return results

    def config_to_dict(self):
        return self.ablation_study.to_dict()

    def log_string(self):
        log = (
            "Maggy Ablation "
            + str(self.result["num_trials"])
            + "/"
            + str(self.num_trials)
            + util._progress_bar(self.result["num_trials"], self.num_trials)
            + " - BEST Excludes"
            + json.dumps(self.result["best_config"])
            + " - metric "
            + str(self.result["best_val"])
        )
        return log
