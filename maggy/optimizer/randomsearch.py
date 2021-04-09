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
import time
from copy import deepcopy

from maggy.optimizer.abstractoptimizer import AbstractOptimizer
from maggy.searchspace import Searchspace


class RandomSearch(AbstractOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_buffer = []

    def initialize(self):

        if (
            Searchspace.DOUBLE not in self.searchspace.names().values()
            and Searchspace.INTEGER not in self.searchspace.names().values()
        ):
            raise NotImplementedError(
                "Searchspace needs at least one continuous parameter for random search."
            )

        self.config_buffer = self.searchspace.get_random_parameter_values(
            self.num_trials
        )

    def get_suggestion(self, trial=None):
        self._log("### start get_suggestion ###")
        self.sampling_time_start = time.time()

        # sampling routine for randomsearch + pruner
        if self.pruner:
            next_trial_info = self.pruner.pruning_routine()
            if next_trial_info == "IDLE":
                self._log(
                    "Worker is IDLE and has to wait until a new trial can be scheduled"
                )
                return "IDLE"
            elif next_trial_info is None:
                # experiment is finished
                self._log("Experiment has finished")
                return None
            elif next_trial_info["trial_id"]:
                # copy hparams of given promoted trial and start new trial with it
                parent_trial_id = next_trial_info["trial_id"]
                parent_trial_hparams = deepcopy(
                    self.get_hparams_dict(trial_ids=parent_trial_id)[parent_trial_id]
                )
                # update trial info dict and create new trial object
                next_trial = self.create_trial(
                    hparams=parent_trial_hparams,
                    sample_type="promoted",
                    run_budget=next_trial_info["budget"],
                )
                self._log("use hparams from promoted trial {}".format(parent_trial_id))
            else:
                # start sampling procedure with given budget
                parent_trial_id = None
                run_budget = next_trial_info["budget"]
                hparams = self.searchspace.get_random_parameter_values(1)[0]
                next_trial = self.create_trial(
                    hparams=hparams, sample_type="random", run_budget=run_budget
                )

            # report new trial id to pruner
            self.pruner.report_trial(
                original_trial_id=parent_trial_id, new_trial_id=next_trial.trial_id
            )

            self._log(
                "start trial {}: {}. info_dict: {} \n".format(
                    next_trial.trial_id, next_trial.params, next_trial.info_dict
                )
            )
            return next_trial

        # sampling routine for pure random search
        elif self.config_buffer:
            run_budget = 0
            next_trial_params = self.config_buffer.pop()
            next_trial = self.create_trial(
                hparams=next_trial_params,
                sample_type="random",
                run_budget=run_budget,
            )

            self._log(
                "start trial {}: {}, {} \n".format(
                    next_trial.trial_id, next_trial.params, next_trial.info_dict
                )
            )

            return next_trial
        else:
            return None

    def finalize_experiment(self, trials):
        return
