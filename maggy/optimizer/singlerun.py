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
from maggy.trial import Trial


class SingleRun(AbstractOptimizer):
    def __init__(self):
        super().__init__()
        self.trial_buffer = []

    def initialize(self):
        for _ in range(self.num_trials):
            self.trial_buffer.append(Trial({}))

    def get_suggestion(self, trial=None):
        if self.trial_buffer:
            return self.trial_buffer.pop()
        else:
            return None

    def finalize_experiment(self, trials):
        return
