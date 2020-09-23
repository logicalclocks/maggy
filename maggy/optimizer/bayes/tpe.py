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

import numpy as np
import statsmodels.api as sm
import scipy.stats as sps

from maggy.optimizer.bayes.base import BaseAsyncBO

"""
The implementation is heavliy inspired by the BOHB (Falkner et al. 2018) paper and the HpBandSter Framework

BOHB: http://proceedings.mlr.press/v80/falkner18a.html
HpBandSter: https://github.com/automl/HpBandSter
"""


class TPE(BaseAsyncBO):
    """TPE based Asynchronous Bayesian Optimization

    TPE always uses Expected Improvement acquisition function, so it is not necessary to specify an acquisition function
    here
    """

    def __init__(
        self,
        gamma=0.15,
        n_samples=24,
        bw_estimation="normal_reference",
        bw_factor=3,
        **kwargs
    ):
        """
        See docstring of `BaseAsyncBO` for more info on parameters of base class

        :param gamma: Determines the percentile of configurations that will be used as training data
                for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
                for training.
        :type gamma: float
        :param n_samples: number of samples drawn from model to optimize EI via sampling
        :type n_samples: int
        :param bw_estimation: method used by statsmodel for the bandwidth estimation of the kde. Options are 'normal_reference', 'silvermann', 'scott'
        :type bw_estimation: str
        :param bw_factor: widens the bandwidth for contiuous parameters for proposed points to optimize EI. Higher values favor more exploration
        :type bw_factor: float
        """
        super().__init__(**kwargs)

        if self.interim_results:
            raise ValueError(
                "Using interim results to update surrogate model is only supported for GP for now, got TPE"
                "Set `interim_results`=False when intitializing the optimizer or use GP"
            )

        # configure tpe specific meta hyperparameters
        self.gamma = gamma
        self.n_samples = n_samples
        self.bw_estimation = bw_estimation
        self.min_bw = 1e-3  # from HpBandSter
        self.bw_factor = bw_factor

    def sampling_routine(self, budget=0):
        best_improvement = -np.inf
        best_sample = None

        kde_good = self.models[budget]["good"]
        kde_bad = self.models[budget]["bad"]

        # alternative way of implementation:
        # instead of this loop, sample all configs beforehand and then optimize acquisition function like in GP
        for sample in range(self.n_samples):
            # randomly choose one of the `good` samples as mean
            idx = np.random.randint(0, len(kde_good.data))
            obs = kde_good.data[idx]
            sample_vector = []

            # loop through hparams
            for mean, bw, hparam_spec in zip(
                obs, kde_good.bw, self.searchspace.items()
            ):

                if hparam_spec["type"] in [
                    self.searchspace.DOUBLE,
                    self.searchspace.INTEGER,
                ]:
                    # sample for cont. hparams
                    # clip by min bw and multiply by factor to favor more exploration
                    bw = max(bw, self.min_bw) * self.bw_factor

                    # low and high are calculated with bounds of hparamsm, because they are always [0,
                    # 1] for transformed hparams we do not have to incorporate them explicitly
                    # `a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std`
                    # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
                    low = -mean / bw
                    high = (1 - mean) / bw

                    rv = sps.truncnorm.rvs(low, high, loc=mean, scale=bw)
                    sample_vector.append(rv)
                else:
                    # sample for categorical hparams (sampling logic taken from HpBandSter)
                    if np.random.rand() < (1 - bw):
                        sample_vector.append(int(mean))
                    else:
                        n_choices = len(hparam_spec["values"])
                        sample_vector.append(np.random.randint(n_choices))

            # calculate EI for current sample
            ei_val = self._calculate_ei(sample_vector, kde_good, kde_bad)

            if ei_val > best_improvement:
                best_improvement = ei_val
                best_sample = sample_vector

        # get original representation of hparams in dict
        best_sample_dict = self.searchspace.list_to_dict(
            self.searchspace.inverse_transform(best_sample)
        )

        return best_sample_dict

    def init_model(self):
        pass

    def update_model(self, budget=0):
        """update surrogate model based on observations from finished trials

            - creating and storing kde for *good* and *bad* observations
            - Only build model when there are more observations than hyperparameters
              i.e. for each kde

        :param budget: the budget for which model should be updated
                       If budget > 0 : multifidelity optimization. Only use observations that were run with
                       `budget` for updateing model for that `budget`. One model exists per budget.
                       If == 0: single fidelity optimization. Only one model exists that is fitted with all observations
        :type budget: int
        """
        # split good and bad trials
        good_hparams, bad_hparams = self._split_trials(budget)

        n_hparams = len(self.searchspace.keys())
        if n_hparams >= len(good_hparams) or n_hparams >= len(bad_hparams):
            self._log(
                "Not enough observations to build model with budget {} yet. n_good_hparams: {}, n_bad_hparams: {}, n_hparmas: {}".format(
                    budget, len(good_hparams), len(bad_hparams), n_hparams
                )
            )
            return

        self._log(
            "Update Model with budget {}. n_good_hparams: {}, n_bad_hparams: {}".format(
                budget, len(good_hparams), len(bad_hparams)
            )
        )

        transformed_good_hparams = np.apply_along_axis(
            self.searchspace.transform, 1, good_hparams
        )
        transformed_bad_hparams = np.apply_along_axis(
            self.searchspace.transform, 1, bad_hparams
        )

        # self._log("good: {}".format(good_hparams))
        # self._log("normalized good: {}".format(transformed_good_hparams))
        # self._log("bad: {}".format(bad_hparams))
        # self._log("normalized bad: {}".format(transformed_bad_hparams))

        var_type = self._get_statsmodel_vartype()

        good_kde = sm.nonparametric.KDEMultivariate(
            data=transformed_good_hparams, var_type=var_type, bw=self.bw_estimation
        )
        bad_kde = sm.nonparametric.KDEMultivariate(
            data=transformed_bad_hparams, var_type=var_type, bw=self.bw_estimation
        )

        self.models[budget] = {"good": good_kde, "bad": bad_kde}

    def _split_trials(self, budget=0):
        """splits trials in good and bad according to tpe algo, for given budget

        We use the logic from the BOHB paper to calculate the best and worst observations.
        This ensures that both models have enough datapoints and have the least overlap when only a limited
        number of observations is available.

        for more details see: https://ml.informatik.uni-freiburg.de/papers/18-ICML-BOHB.pdf

        :param budget: the budget for which observations shoul be split
        :type budget: int
        :return: tuple with arrays of good trials and bad trials
        :rtype (np.ndarray(n_trials, n_hparams), np.ndarray(n_trials, n_params))
        """

        metric_history = self.get_metrics_array(budget=budget)
        metric_idx_ascending = np.argsort(metric_history)
        hparam_history = self.get_hparams_array(budget=budget)

        n_good = max(
            len(self.searchspace.keys()) + 1, int(self.gamma * metric_history.shape[0])
        )
        n_bad = max(
            len(self.searchspace.keys()) + 1,
            int((1 - self.gamma) * metric_history.shape[0]),
        )

        good_trials = hparam_history[metric_idx_ascending[:n_good]]
        bad_trials = hparam_history[metric_idx_ascending[n_good : n_good + n_bad]]

        return good_trials, bad_trials

    def _get_statsmodel_vartype(self):
        """Returns *statsmodel* type specifier string consisting of the types for each hparam of the searchspace , so for example 'ccuo'.

        :rtype: str
        """

        var_type_string = ""
        for hparam_spec in self.searchspace.items():
            var_type_string += TPE._get_vartype(hparam_spec["type"])

        return var_type_string

    @staticmethod
    def _get_vartype(maggy_vartype):
        """Transforms Maggy vartype to statsmodel vartype, e.g. 'DOUBLE' → 'c'

        :param maggy_vartype: maggy type of hparam, e.g. 'DOUBLE'
        :type maggy_vartype: str
        :returns: corresponding vartype of statsmodel
        :rtype: str
        """
        if maggy_vartype == "DOUBLE":
            return "c"
        elif maggy_vartype == "INTEGER":
            return "c"
        elif maggy_vartype == "CATEGORICAL":
            return "u"
        else:
            raise NotImplementedError("Only cont vartypes are implemented yer")

    @staticmethod
    def _calculate_ei(x, kde_good, kde_bad):
        """Returns Expected Improvement for given hparams

        :param x: list of hyperparameters, shape(n_hparams,)
        :type x: list
        :param kde_good: kde of good observations
        :type kde_good: sm.KDEMultivariate
        :param kde_bad: pdf of kde of bad observations
        :type kde_bad: sm.KDEMultivariate of KDE instance
        :return: expected improvement
        :rtype: float
        """
        return max(1e-32, kde_good.pdf(x)) / max(kde_bad.pdf(x), 1e-32)
