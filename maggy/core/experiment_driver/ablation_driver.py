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
from typing import Callable, Union, Optional

from maggy import util
from maggy.config import AblationConfig
from maggy.earlystop import NoStoppingRule
from maggy.ablation.ablationstudy import AblationStudy
from maggy.ablation.ablator.loco import LOCO
from maggy.ablation.ablator import AbstractAblator
from maggy.trial import Trial
from maggy.core.rpc import OptimizationServer
from maggy.core.experiment_driver.optimization_driver import HyperparameterOptDriver
from maggy.core.executors.trial_executor import trial_executor_fn


class AblationDriver(HyperparameterOptDriver):
    """Driver class for ablation experiments.

    Initializes a controller that returns a given network with a new ablated
    feature from the searchspace on each poll. Implements the experiment driver
    callbacks.
    """

    def __init__(self, config: AblationConfig, app_id: int, run_id: int):
        """Performs argument checks and initiallizes the ablation controller.

        :param config: Experiment config.
        :param app_id: Maggy application ID.
        :param run_id: Maggy run ID.

        :raises TypeError: If the ablation policy or ablation study config is
            wrong.
        """
        super().__init__(config, app_id, run_id)
        # set up an ablation study experiment
        self.earlystop_check = NoStoppingRule.earlystop_check

        if isinstance(config.ablation_study, AblationStudy):
            self.ablation_study = config.ablation_study
        else:
            raise TypeError(
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
            raise TypeError(
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

    def _exp_startup_callback(self) -> None:
        """No special startup actions required."""

    def _exp_final_callback(self, job_end: int, exp_json: dict) -> dict:
        """Writes the results from the ablation study into a dict and logs it.

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
            "N/A",
        )
        print("Finished experiment.")
        return result

    def _exp_exception_callback(self, exc: Exception) -> None:
        """Raises the driver exception if existent, else reraises unhandled
        exception.
        """
        if self.exception:
            raise self.exception  # pylint: disable=raising-bad-type
        raise exc

    def _patching_fn(self, train_fn: Callable, config: AblationConfig) -> Callable:
        """Monkey patches the user training function with the trial executor
        modifications for ablation studies.

        :param train_fn: User provided training function.

        :returns: The monkey patched training function."""
        return trial_executor_fn(
            train_fn,
            config,
            "ablation",
            self.app_id,
            self.run_id,
            self.server_addr,
            self.hb_interval,
            self._secret,
            "N/A",
            self.log_dir,
        )

    def controller_get_next(self, trial: Optional[Trial] = None) -> Union[Trial, None]:
        """Gets a `Trial` to be assigned to an executor, or `None` if there are
        no trials remaining in the experiment.

        :param trial: Trial to fetch from the controller (default ``None``).
            None autofetches the next available trial.
        """
        return self.controller.get_trial(trial)

    def prep_results(self, duration_str: str) -> str:
        """Writes and returns the results of the experiment into one string and
        returns it.

        :param duration_str: Experiment duration as a formatted string.

        :returns: The formatted experiment results summary string.
        """
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

    def config_to_dict(self) -> dict:
        """Returns a summary of the scheduled ablation study as a dict.

        :returns: The summary dict.
        """
        return self.ablation_study.to_dict()

    def log_string(self) -> str:
        """Returns a log string for the progress bar in jupyter.

        :returns: The progress string.
        """
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
