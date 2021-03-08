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

import json

from maggy import util
from maggy.earlystop import NoStoppingRule
from maggy.ablation.ablationstudy import AblationStudy
from maggy.ablation.ablator.loco import LOCO
from maggy.ablation.ablator import AbstractAblator
from maggy.core.rpc import OptimizationServer
from maggy.core.experiment_driver.optimization_driver import OptimizationDriver
from maggy.core.executors.trial_executor import trial_executor_fn


class AblationDriver(OptimizationDriver):
    def __init__(self, config, app_id, run_id):
        super().__init__(config, app_id, run_id)
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
        self.server = OptimizationServer(self.num_executors)
        self.result = {"best_val": "n.a.", "num_trials": 0, "early_stopped": "n.a"}

        # Init controller and set references to data in ablator
        self.direction = config.direction
        self.controller.ablation_study = self.ablation_study
        self.controller.final_store = self._final_store
        self.controller.initialize()

    def _exp_startup_callback(self):
        pass

    def _exp_final_callback(self, job_end, exp_json):
        result = self.finalize(job_end)
        best_logdir = self.log_dir + "/" + result["best_id"]
        util.finalize_experiment(
            exp_json,
            float(result["best_val"]),
            self.APP_ID,
            self.RUN_ID,
            "FINISHED",
            self.duration,
            self.log_dir,
            best_logdir,
            "N/A",
        )
        print("Finished experiment.")
        return result

    def _exp_exception_callback(self, exc):
        if self.exception:
            raise self.exception  # pylint: disable=raising-bad-type
        raise exc

    def _patching_fn(self, train_fn):
        return trial_executor_fn(
            train_fn,
            "ablation",
            self.APP_ID,
            self.RUN_ID,
            self.server_addr,
            self.hb_interval,
            self._secret,
            "N/A",
            self.log_dir,
        )

    def controller_get_next(self, trial=None):
        return self.controller.get_trial(trial)

    def prep_results(self, duration_str):
        self.controller.finalize_experiment(self._final_store)
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
            + duration_str
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
            + util.progress_bar(self.result["num_trials"], self.num_trials)
            + " - BEST Excludes"
            + json.dumps(self.result["best_config"])
            + " - metric "
            + str(self.result["best_val"])
        )
        return log
