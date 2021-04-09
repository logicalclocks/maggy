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

import itertools

from maggy import Searchspace
from maggy.optimizer.abstractoptimizer import AbstractOptimizer


class GridSearch(AbstractOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_buffer = []

    def initialize(self):
        self._validate_searchspace(self.searchspace)
        # create all trials ahead of time
        self.config_buffer = self._grid_params(self.searchspace)

    @classmethod
    def get_num_trials(cls, searchspace):
        """For grid search the number of trials is determined by the size of the
        cartisian product, depending on the user-set number of parameters and values

        This method is duplicating part of the code in the `initialize()` mainly to keep
        the flow of things the same as for other optimizers, where the user sets only
        the number of trials to evaluate.
        """
        cls._validate_searchspace(searchspace)
        return len(cls._grid_params(searchspace))

    def get_suggestion(self, trial=None):
        # sampling routine for randomsearch + pruner
        if self.pruner:
            raise NotImplementedError(
                "Grid search in combination with trial pruning "
                "is currently not supported."
            )
        elif self.config_buffer:
            run_budget = 0
            next_trial_params = self.config_buffer.pop()
            next_trial = self.create_trial(
                hparams=next_trial_params,
                sample_type="grid",
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

    @staticmethod
    def _grid_params(searchspace):
        return_list = []
        for hparams in itertools.product(
            *[item["values"] for item in searchspace.items()]
        ):
            return_list.append(searchspace.list_to_dict(hparams))
        return return_list

    @staticmethod
    def _validate_searchspace(searchspace):
        if (
            Searchspace.DOUBLE in searchspace.names().values()
            or Searchspace.INTEGER in searchspace.names().values()
        ):
            raise NotImplementedError(
                "Searchspace can only contain `discrete` or `categorical` "
                "hyperparameters for grid search."
            )
