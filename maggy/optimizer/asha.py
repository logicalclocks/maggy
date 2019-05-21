import math

from maggy.optimizer import AbstractOptimizer
from maggy.searchspace import Searchspace
from maggy.trial import Trial

"""we need additional parameters that is the promotion rate and the max
resource constraint.
"""


class Asha(AbstractOptimizer):

    def initialize(self):
        self.reduction_factor = self.searchspace.get('reduction_factor', None)
        if self.reduction_factor is None:
            raise Exception(
                "Can't initialize ASHA optimizer without 'reduction_factor' \
                parameter in Searchspace.")
        else:
            if (self.reduction_factor[0]
                not in [Searchspace.DISCRETE, Searchspace.CATEGORICAL]):
                raise Exception(
                    "Can't initialize ASHA optimizer. 'reduction_factor' \
                    not of type DISCRETE or CATEGORICAL.")
            elif len(self.reduction_factor[1]) > 1:
                raise Exception(
                    "Can't initialize ASHA optimizer. 'reduction_factor' \
                    can only be a single value: {}"
                    .format(self.reduction_factor[1]))
            elif self.reduction_factor[1][0] < 2 or not isinstance(self.reduction_factor[1][0], int):
                raise Exception(
                    "Can't initialize ASHA optimizer. 'reduction_factor' \
                    has to be an integer equal to or larger than 2: {}"
                    .format(self.reduction_factor[1]))
            else:
                self.reduction_factor = self.reduction_factor[1][0]

        self.resource = self.searchspace.get('resource', None)

        if self.resource is None:
            raise Exception(
                "Can't initialize ASHA optimizer without 'resource' \
                parameter in Searchspace.")
        else:
            if self.resource[0] != Searchspace.INTEGER:
                raise Exception(
                    "Can't initialize ASHA optimizer. 'resource' \
                    not of type INTEGER.")
            elif len(self.resource[1]) > 1:
                raise Exception(
                    "Can't initialize ASHA optimizer. 'resource' \
                    can only be a single value: {}"
                    .format(self.resource[1]))
            else:
                self.resource = self.resource[1]
                self.resource_min = min(self.resource)
                self.resource_max = max(self.resource)

        # maps rung index k to trials in that rung
        self.rungs = {0: []}
        # maps rung index k to trial ids of trials that were promoted
        self.promoted = {0: []}

        self.max_rung = int(math.floor(math.log(
            self.resource_max/self.resource_min, self.reduction_factor)))


    def get_suggestion(self, trial=None):

        if trial is not None:
            # stopping criterium: one trial in max rung
            if self.max_rung in self.rungs:
                # return None to signal end to experiment driver
                return None

            # for each rung
            for k in range(self.max_rung, -1, -1):
                # if rung doesn't exist yet go one lower
                if k not in self.rungs:
                    continue
                # get top_k
                candidates = self._top_k(k, len(self.rungs[k])//self.reduction_factor)
                if not candidates:
                    continue
                # select all that haven't been promoted yet in top_k
                promotable = [t for t in candidates if t.trial_id not in self.promoted[k]]

                nr_promotable = len(promotable)
                if nr_promotable == 1:
                    new_rung = k + 1
                    t = promotable[0]
                    params = t.params
                    params.pop('reduction_factor', None)
                    params['resource'] = self.resource_min * (self.reduction_factor**new_rung)
                    promote_trial = Trial(params)
                    if new_rung in self.rungs:
                        self.rungs[new_rung].append(promote_trial)
                    else:
                        self.rungs[new_rung] = [promote_trial]

                    if new_rung in self.promoted:
                        self.promoted[k].append(promote_trial.trial_id)
                    else:
                        self.promoted[k] = [promote_trial.trial_id]

                    return promote_trial
                elif nr_promotable > 1:
                    raise Exception("More than one trial promotable")

            # return random configuration in base rung
            # get one random combination
            params = self.searchspace.get_random_parameter_values(1)
            # set resource to minimum
            params.pop('reduction_factor', None)
            params['resource'] = self.resource_min
            trial = Trial(params)
            self.rungs[0].append(trial)
            return trial

    def finalize_experiment(self, trials):
        return

    def _top_k(self, rung_k, number):
        if number > 0:
            self.rungs[rung_k].sort(key=lambda x: x.final_metric, reverse=True)
            return self.rungs[rung_k][:number]
        else:
            return []
