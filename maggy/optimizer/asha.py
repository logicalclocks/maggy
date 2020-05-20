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

import math

from maggy.optimizer.abstractoptimizer import AbstractOptimizer
from maggy.trial import Trial


class Asha(AbstractOptimizer):
    """Implements the Asynchronous Successiv Halving Algorithm - ASHA
    (https://arxiv.org/abs/1810.05934). ASHA needs three additional parameters:
    'reduction_factor', 'resource_min' and 'resource_max'. To set custom values
    for these, initialize the optimizer first and pass it as an argument to
    'experiment.lagom()'.

    Sample usage:

    >>> # Import Asha optimizer
    >>> from maggy.optimizer import Asha
    >>> # Instantiate the optimizer with custom arguments
    >>> asha = Asha(3, 1, 9)
    >>> experiment.lagom(..., optimizer=asha, ...)
    """

    def __init__(self, reduction_factor=2, resource_min=1, resource_max=4):
        super().__init__()

        if reduction_factor < 2 or not isinstance(reduction_factor, int):
            raise Exception(
                "Can't initialize ASHA optimizer. 'reduction_factor'"
                + "has to be an integer equal to or larger than 2: {}".format(
                    reduction_factor
                )
            )
        else:
            self.reduction_factor = reduction_factor

        if not isinstance(resource_min, int):
            raise Exception(
                "Can't initialize ASHA optimizer. 'resource_min'"
                + "not of type INTEGER."
            )
        if not isinstance(resource_max, int):
            raise Exception(
                "Can't initialize ASHA optimizer. 'resource_max'"
                + "not of type INTEGER."
            )
        if resource_min >= resource_max:
            raise Exception(
                "Can't initialize ASHA optimizer. 'resource_min' is larger"
                + "than 'resource_max'."
            )

        self.resource_min = resource_min
        self.resource_max = resource_max

    def initialize(self):

        # maps rung index k to trials in that rung
        self.rungs = {0: []}
        # maps rung index k to trial ids of trials that were promoted
        self.promoted = {0: []}

        self.max_rung = int(
            math.floor(
                math.log(self.resource_max / self.resource_min, self.reduction_factor)
            )
        )

        assert self.num_trials >= self.reduction_factor ** (self.max_rung + 1)

    def get_suggestion(self, trial=None):

        if trial is not None:
            # stopping criterium: one trial in max rung
            if self.max_rung in self.rungs:
                # return None to signal end to experiment driver
                return None

            # for each rung
            for k in range(self.max_rung - 1, -1, -1):
                # if rung doesn't exist yet go one lower
                if k not in self.rungs:
                    continue

                # get top_k
                rung_finished = len(
                    [x for x in self.rungs[k] if x.status == Trial.FINALIZED]
                )

                if (rung_finished // self.reduction_factor) - len(
                    self.promoted.get(k, [])
                ) > 0:
                    candidates = self._top_k(
                        k, (rung_finished // self.reduction_factor)
                    )
                else:
                    candidates = []

                # if there are no candidates, check one rung below
                if not candidates:
                    continue

                # select all that haven't been promoted yet
                promotable = [
                    t for t in candidates if t.trial_id not in self.promoted.get(k, [])
                ]

                nr_promotable = len(promotable)
                if nr_promotable >= 1:
                    new_rung = k + 1
                    # sorted in decending order, take highest -> index 0
                    old_trial = promotable[0]
                    # make copy of params to be able to change resource
                    params = old_trial.params.copy()
                    params["resource"] = self.resource_min * (
                        self.reduction_factor ** new_rung
                    )
                    promote_trial = Trial(params)

                    # open new rung if not exists
                    if new_rung in self.rungs:
                        self.rungs[new_rung].append(promote_trial)
                    else:
                        self.rungs[new_rung] = [promote_trial]

                    # remember promoted trial
                    if k in self.promoted:
                        self.promoted[k].append(old_trial.trial_id)
                    else:
                        self.promoted[k] = [old_trial.trial_id]

                    return promote_trial

        # else return random configuration in base rung
        params = self.searchspace.get_random_parameter_values(1)[0]
        # set resource to minimum
        params["resource"] = self.resource_min
        to_return = Trial(params)
        # add to bottom rung
        self.rungs[0].append(to_return)
        return to_return

    def finalize_experiment(self, trials):
        return

    def _top_k(self, rung_k, number):
        """Find top-`number` trials in `rung_k`.
        """
        if number > 0:
            filtered = [x for x in self.rungs[rung_k] if x.status == Trial.FINALIZED]
            filtered.sort(key=lambda x: x.final_metric, reverse=True)
            # return top k trials if finalized
            return filtered[:number]
        else:
            return []
