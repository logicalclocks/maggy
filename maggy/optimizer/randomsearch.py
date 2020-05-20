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

from maggy.optimizer.abstractoptimizer import AbstractOptimizer
from maggy.searchspace import Searchspace
from maggy.trial import Trial


class RandomSearch(AbstractOptimizer):
    def __init__(self):
        super().__init__()
        self.trial_buffer = []

    def initialize(self):

        if (
            Searchspace.DOUBLE not in self.searchspace.names().values()
            and Searchspace.INTEGER not in self.searchspace.names().values()
        ):
            raise NotImplementedError(
                "Searchspace needs at least one continuous parameter for random search."
            )

        list_of_random_trials = self.searchspace.get_random_parameter_values(
            self.num_trials
        )
        for parameters_dict in list_of_random_trials:
            self.trial_buffer.append(Trial(parameters_dict, trial_type="optimization"))

    def get_suggestion(self, trial=None):
        if self.trial_buffer:
            return self.trial_buffer.pop()
        else:
            return None

    def finalize_experiment(self, trials):
        return
