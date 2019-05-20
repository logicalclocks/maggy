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

        self.rungs = {}


    def get_suggestion(self, trial=None):
        # first couple of trials just get random trial from base rung
        if trial is None:
            # get one random combination
            params = self.searchspace.get_random_parameter_values(1)
            # set resource to minimum
            params.pop('reduction_factor', None)
            params['resource'] = self.resource_min
            return Trial(params)
        if trial is not None:
            


    def finalize_experiment(self, trials):
        return

    def _get_job(self):
        pass

    def _top_k(self, rung_k, number):
        pass
