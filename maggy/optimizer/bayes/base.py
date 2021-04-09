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
from abc import abstractmethod

import numpy as np

from maggy.optimizer.abstractoptimizer import AbstractOptimizer


class BaseAsyncBO(AbstractOptimizer):
    """Base class for asynchronous bayesian optimization

    Do not initialize this class! only it's subclasses

    **async bo**

    Bayesian Optimization consists of a surrogat model to approximate the black-box function and an acquisition
    function to select the next hyperparameter configuration. In an asynchronous setting, we need to encourage diversity
    when maximizing the acquisition function to prevent redundant sampling.

    **optimization direction**

    all optimizations are converted to minimizations problems via the `get_metrics_dict()` and `get_metrics_array()`
    methods, where the metrics are negated in case of a maximization.
    In the `final_store` the original metric values are saved.

    **pruner**

    Pruners can be run as a subroutine of an optimizer. It's interface `pruner.pruning_routine()` is called at the beginning
    of the `get_suggestion()` method and returns the hparam config and budget for the trial. The `trial_id` of the
    newly created Trial object is reported back to the pruner via `pruner.report_trial()`.

    **budget**

    If a pruner was specified ( and hence a multi-fidelity optimization is conducted ), the budget is added as an
    additional hyperparameter to the hparam dict so it is passed to the map_fun. Note, that when fitting the
    surrogate model the "budget" key is obviously omitted.

    **models**

    The surrogate models for different budgets are saved in the `models` dict with the budgets as key. In case of
    a single fidelity optimization without a pruner or pruner with interim_results, The only model has the key `0`.
    If we fit multiple models, sample new hparam configs always from the largest model available (i.e. biggest budget)

    **imputing busy trials** (poss. shift to simple.py)

    the imputed metrics are calculated on the fly in `get_metrics_array(include_busy_locations=True, budget=`budget`)`
    for currently evaluating trials that were sampled from model with `budget`

    **interim results**

    In literature most surrogate models of BO algorithms are fit with the hyperparameter config and the final metric
    of a trial.
    Using additionally interim metrics of trials, i.e. use metrics that were trained for varying budgets makes it possible
    to use early stopping in BO. (Also it makes it possible to have only one model in multi fidelity bo instead of one
    model per fidelity)
    For each interim metric a hyperparameter config augumented with the budget the metric was generated with is added
    to the training data of the surrogate model

    .. math::
                                    z_t = [x_t, n_t] ; y_t = y_{t,nt}

    When maximizing the acquisition function to sample the next hyperparameter config to evaluate, always augument with
    the max budget N

    .. math::
                                    xt ← argmax acq([x, N])


    """

    def __init__(
        self,
        num_warmup_trials=15,
        random_fraction=0.33,
        interim_results=False,
        interim_results_interval=10,
        **kwargs
    ):
        """
        Attributes
        ----------

        models (dict): The surrogate models for different budgets are saved in the `models` dict with the budgets as key.
                       In case of a single fidelity optimization without a pruner. The only model has the key `0`.
        warm_up_configs(list[dict]): list of hparam configs used for warming up
        sampling_time_start (float): helper variable to calculate time needed for calculating next suggestion, i.e
                                       when sampling from model (`sample_type`=="model"). Calulating the time happens
                                       in `create_trial()`
        normalize_categorical (bool): If True, the encoded categorical hparam is also max-min
                                      normalized between 0 and 1 in searchspace.transform()

        :param num_warmup_trials: number of random trials at the beginning of experiment
        :type num_warmup_trials: int
        :param random_fraction: fraction of random samples, between [0,1]
        :type random_fraction: float
        :param interim_results: If True, use interim metrics from trials for fitting surrogate model. Else use final
                                metrics only
        :type interim_results: bool
        :param interim_results_interval: Specifies which interim metrics are used (if interim_results==True)
                                         e.g. interval=10: the metric of every 10th epoch is used for fitting surrogate
        :type interim_results_interval: int
        """
        super().__init__(**kwargs)

        # configure warmup routine
        self.num_warmup_trials = num_warmup_trials
        self.warmup_sampling = "random"
        self.warmup_configs = []  # keeps track of warmup warmup configs

        allowed_warmup_sampling_methods = ["random"]
        if self.warmup_sampling not in allowed_warmup_sampling_methods:
            raise ValueError(
                "expected warmup_sampling to be in {}, got {}".format(
                    allowed_warmup_sampling_methods, self.warmup_sampling
                )
            )

        # surrogate model related aruments
        self.models = {}  # fitted model of the estimator
        self.random_fraction = random_fraction
        self.interim_results = interim_results
        self.interim_results_interval = interim_results_interval

        # helper variable to calculate time needed for calculating next suggestion
        self.sampling_time_start = 0.0

        # If True, the encoded categorical hparam is also max-min normalized between 0 and 1 in searchspace.transform()
        self.normalize_categorical = True
        if self.name() == "TPE":
            self.normalize_categorical = False

    def initialize(self):
        # validate hparam types
        # at least one hparam needs to be continuous & no DISCRETE hparams
        cont = False
        for hparam in self.searchspace.items():
            if hparam["type"] == self.searchspace.DISCRETE:
                raise ValueError(
                    "This version of Bayesian Optimization does not support DISCRETE Hyperparameters yet, please encode {} as INTEGER".format(
                        hparam["name"]
                    )
                )
            if hparam["type"] in [self.searchspace.DOUBLE, self.searchspace.INTEGER]:
                cont = True
        if not cont:
            raise ValueError(
                "In this version of Bayesian Optimization at least one hparam has to be continuous (DOUBLE or INTEGER)"
            )

        self.warmup_routine()
        self.init_model()

    def get_suggestion(self, trial=None):
        self._log("### start get_suggestion ###")
        self.sampling_time_start = time.time()
        if trial:
            self._log(
                "last finished trial: {} with params {}".format(
                    trial.trial_id, trial.params
                )
            )
        else:
            self._log("no previous finished trial")

        # check if experiment has finished
        if self._experiment_finished():
            return None

        # pruning routine
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
                # report new trial id to pruner
                self.pruner.report_trial(
                    original_trial_id=parent_trial_id,
                    new_trial_id=next_trial.trial_id,
                )
                self._log(
                    "use hparams from promoted trial {}. \n start trial {}: {} \n".format(
                        parent_trial_id, next_trial.trial_id, next_trial.params
                    )
                )
                return next_trial
            else:
                # start sampling procedure with given budget
                run_budget = next_trial_info["budget"]
                if self.interim_results:
                    model_budget = 0
                else:
                    model_budget = run_budget
        else:
            run_budget = 0
            model_budget = 0

        # check if there are still trials in the warmup buffer
        if self.warmup_configs:
            self._log("take sample from warmup buffer")
            next_trial_params = self.warmup_configs.pop()
            next_trial = self.create_trial(
                hparams=next_trial_params,
                sample_type="random",
                run_budget=run_budget,
            )

        elif np.random.rand() < self.random_fraction:
            # random fraction applies, sample randomly
            hparams = self.searchspace.get_random_parameter_values(1)[0]
            next_trial = self.create_trial(
                hparams=hparams, sample_type="random", run_budget=run_budget
            )
            self._log("sampled randomly: {}".format(hparams))
        else:
            # update model
            if self.pruner and not self.interim_results:
                # skip model building if we already have a bigger model
                if max(list(self.models.keys()) + [-np.inf]) <= model_budget:
                    self.update_model(model_budget)
            else:
                self.update_model(model_budget)

            if not self.models:
                # in case there is no model yet, sample randomly
                hparams = self.searchspace.get_random_parameter_values(1)[0]
                next_trial = self.create_trial(
                    hparams=hparams, sample_type="random", run_budget=run_budget
                )
                self._log("sampled randomly: {}".format(hparams))
            else:
                if self.pruner and not self.interim_results:
                    # sample from largest model available
                    model_budget = max(self.models.keys())
                # sample from model with model budget
                self._log(
                    "start sampling routine from model with budget {}".format(
                        model_budget
                    )
                )
                hparams = self.sampling_routine(model_budget)
                next_trial = self.create_trial(
                    hparams=hparams,
                    sample_type="model",
                    run_budget=run_budget,
                    model_budget=model_budget,
                )
                self._log(
                    "sampled from model with budget {}: {}".format(
                        model_budget, hparams
                    )
                )

        # check if Trial with same hparams has already been created
        i = 0
        while self.hparams_exist(trial=next_trial):
            self._log("sample randomly to encourage exploration")
            hparams = self.searchspace.get_random_parameter_values(1)[0]
            next_trial = self.create_trial(
                hparams=hparams, sample_type="random_forced", run_budget=run_budget
            )
            i += 1
            if i > 3:
                self._log(
                    "not possible to sample new config. Stop Experiment (most/all configs have already been used)"
                )
                return None

        # report new trial id to pruner
        if self.pruner:
            self.pruner.report_trial(
                original_trial_id=None, new_trial_id=next_trial.trial_id
            )
        self._log(
            "start trial {}: {}, {} \n".format(
                next_trial.trial_id, next_trial.params, next_trial.info_dict
            )
        )
        return next_trial

    def finalize_experiment(self, trials):
        return

    @abstractmethod
    def init_model(self):
        """initializes the surrogate model of the gaussian process

        the model gets created with the right parameters, but is not fit with any data yet. the `base_model` will be
        cloned in `update_model` and fit with observation data
        """
        raise NotImplementedError

    @abstractmethod
    def update_model(self, budget=0):
        """update surrogate model with new observations

        Use observations of finished trials + liars from busy trials to build model.
        Only build model when there are at least as many observations as hyperparameters

        :param budget: the budget for which model should be updated. Default is 0
                       If budget > 0 : multifidelity optimization. Only use observations that were run with
                       `budget` for updateing model for that `budget`. One model exists per budget.
                       If == 0: single fidelity optimization. Only one model exists that is fitted with all observations
        :type budget: int
        """
        raise NotImplementedError

    @abstractmethod
    def sampling_routine(self, budget=0):
        """Samples new config from model

        This methods holds logic for:

        - maximizing acquisition function based on current model and observations
        - async logic: i.e. imputing busy_locations with a liar to encourage diversity in sampling

        :param budget: the budget from which model should be sampled. Default is 0
                       If budget > 0 : multifidelity optimization. Only use observations that were run with
                       `budget` for updateing model for that `budget`. One model exists per budget.
                       If == 0: single fidelity optimization. Only one model exists that is fitted with all observations
        :type budget: int
        :return: hyperparameter config that minimizes the acquisition function
        :rtype: dict
        """
        raise NotImplementedError

    def warmup_routine(self):
        """implements logic for warming up bayesian optimization through random sampling by adding hparam configs to
        `warmup_config` list
        """

        # generate warmup hparam configs
        if self.warmup_sampling == "random":
            self.warmup_configs = self.searchspace.get_random_parameter_values(
                self.num_warmup_trials
            )
        else:
            raise NotImplementedError(
                "warmup sampling {} doesnt exist, use random".format(
                    self.warmup_sampling
                )
            )

    def _experiment_finished(self):
        """checks if experiment is finished

        In normal BO, experiment has finished when specified amount of trials have run,
        in BOHB/ASHA when all iterations have been finished

        :return: True if experiment has finished, False else
        :rtype: bool
        """
        if self.pruner:
            if self.pruner.finished():
                return True
        elif len(self.final_store) >= self.num_trials:
            self._log(
                "Finished experiment, ran {}/{} trials".format(
                    len(self.final_store), self.num_trials
                )
            )
            return True
        else:
            return False

    def get_busy_locations(self, budget=0):
        """returns hparams of currently evaluating trials

        Considers only trials that were sampled from model with specified budget
        This is a helper functions used when async strategy is `impute`
        """
        if not self.include_busy_locations():
            raise ValueError(
                "Only Optimizer GP with async_strategy == `impute` can include busy locations. Got Optimizer {} {}".format(
                    self.name(),
                    "async_strategy: " + self.async_strategy
                    if self.name() == "GP"
                    else "",
                )
            )

        hparams_busy = np.array(
            [
                self.searchspace.dict_to_list(trial.params)
                for trial_id, trial in self.trial_store.items()
                if trial.info_dict["sample_type"] == "model"
                and trial.info_dict["model_budget"] == budget
            ]
        )

        return hparams_busy

    def get_imputed_metrics(self, budget=0):
        """returns imputed metrics for currently evaluating trials

        Considers only trials that were sampled from model with specified budget
        This is a helper function, only used when async strategy is `impute`
        """
        if not self.include_busy_locations():
            raise ValueError(
                "Only Optimizer GP with async_strategy == `impute` can include busy locations. Got Optimizer {} {}".format(
                    self.name(),
                    "async_strategy: " + self.async_strategy
                    if self.name() == "GP"
                    else "",
                )
            )

        metrics_busy = np.empty(0, dtype=float)
        for trial_id, trial in self.trial_store.items():
            if (
                trial.info_dict["sample_type"] == "model"
                and trial.info_dict["model_budget"] == budget
            ):
                imputed_metric = self.impute_metric(trial.params, budget)
                metrics_busy = np.append(metrics_busy, imputed_metric)
                # add info about imputed metric to trial info dict
                if "imputed_metrics" in trial.info_dict.keys():
                    trial.info_dict["imputed_metrics"].append(imputed_metric)
                else:
                    trial.info_dict["imputed_metrics"] = [imputed_metric]

        return metrics_busy

    def get_XY(self, budget=0, interim_results=False, interim_results_interval=10):
        """get transformed hparams and metrics for fitting surrogate

        :param budget: budget for which model should be build. Default is 0
                       If budget > 0 : One model exists per budget. Only use observations that were run with
                       `budget` for updateing model for that `budget`.
                       If == 0: Only one model exists that is fitted with all observations
        :type budget: int
        :param interim_results: If True interim results from metric history are used and hparams are augumented with budget.
                                i.e. for every interim result have one observation with
                                .. math::
                                    z_t = [x_t, n_t] ; y_t = y_{t,nt}
                                Else use final metrics only
        :type interim_results: bool
        :param interim_results_interval: Specifies the interval of the interim results being used, If interim_results==True
                                        e.g. if 10, use every metric of every 10th epoch
        :type interim_results_interval: int
        :return: Tuple of 2 arrays, the first containing hparams and metrics.
                 There are four scenarios:

                 * Hparams and Metrics of finalized trials
                   shapes: (n_finalized_trials, n_hparams), (n_finalized_trials,)
                 * Hparams and Metrics of finalized trials
                   + hparams and imputed metrics of busy_locations (if async_strategy=="asnyc")
                   shapes: (n_finalized_trials+n_busy_locations, n_hparams), (n_finalized_trials+n_busy_locations,)
                 * Hparams (augumented with budget) and interim metrics of finalized trials
                   shapes: (n_interim_results, n_hparams + 1), (n_interim_results,)
                           Note that first and final metric of each trial are always used
                           and that trials may be trained with different budgets
                 * Hparams (augumented with budget) and interim metrics of finalized trials
                   + hparams and imputed final metric for evaluating trials (if async_strategy=="asnyc")
                   shapes: (n_interim_results+n_busy_locations, n_hparams + 1),
                           (n_interim_results+n_busy_locations,)

        :rtype: (np.ndarray, np.ndarray)
        """
        if not interim_results:
            # return final metrics only

            # get hparams and final metrics of finalized trials
            hparams = self.get_hparams_array(budget=budget)
            metrics = self.get_metrics_array(budget=budget, interim_metrics=False)

            # if async strategy is `impute`
            if self.include_busy_locations():
                # get hparams and imputed metrics of evaluating trials
                hparams_busy = self.get_busy_locations(budget=budget)
                imputed_metrics = self.get_imputed_metrics(budget=budget)
                assert hparams_busy.shape[0] == imputed_metrics.shape[0], (
                    "Number of evaluating trials and imputed "
                    "metrics needs to be equal, "
                    "got n_busy_locations: {}, "
                    "n_imputed_metrics: {}".format(
                        hparams_busy.shape[0], imputed_metrics.shape[0]
                    )
                )
                # append to hparams and metrics
                if len(hparams_busy) > 0:
                    hparams = np.concatenate((hparams, hparams_busy))
                    metrics = np.concatenate((metrics, imputed_metrics))

            # transform hparams
            # note that through transform, budget param gets ommited from hparams if it was existent (pruner)
            hparams_transform = np.apply_along_axis(
                self.searchspace.transform,
                1,
                hparams,
                normalize_categorical=self.normalize_categorical,
            )

            X = hparams_transform
            y = metrics

            assert X.shape[1] == len(
                self.searchspace.keys()
            ), "shape[1] of X needs to be equal to number of hparams"

        elif interim_results:
            # so far only supported for GP and no pruner

            # return interim results and hparams augumented with budget
            # return every nth interim result according to interim_results_interval. always return first and last result

            # get and transform hparams of all finalized trials
            hparams = self.get_hparams_array(budget=budget)
            hparams_transform = np.apply_along_axis(
                self.searchspace.transform,
                1,
                hparams,
                normalize_categorical=self.normalize_categorical,
            )

            # get full metric history for all finalized trials
            metrics = self.get_metrics_array(
                interim_metrics=True, budget=budget
            )  # array of metric history arrays

            # get indices of hparams/metrics to be used for each trial
            interim_result_indices = np.array(
                [
                    self.get_interim_result_idx(
                        metric_history, interim_results_interval
                    )
                    for metric_history in metrics
                ]
            )  # araay of list. each list represents the indices of the metrics to be used of that trial

            # only use every nth interim result of metric history. specified with interim_result_indices
            metrics_filtered = np.array(
                [
                    metrics[trial_idx][interim_result_indices[trial_idx]]
                    for trial_idx in range(metrics.shape[0])
                ]
            )
            # flatten results so they can be used for fitting posterior
            # if arrays in yi_filtered do not have same length, e.g. some trials were early stopped or pruner was activated,
            # yi_filtered is ragged (yi_filtered.shape = (n_trials,)) and flatten does not work, use np.hstack().
            # if all trials have been trained on max budget (yi_filtered.shape = (n_trials, n_filtered_results)) use flatten() since it is better for performance.
            if len(metrics_filtered.shape) == 2:
                metrics_flat = metrics_filtered.flatten()
            else:
                metrics_flat = np.hstack(metrics_filtered)

            # augument hparams with budget, i.e. z_t = [x_t, n_t] for every interim result
            max_budget = self.get_max_budget()
            n_hparams = len(self.searchspace.keys())
            hparams_augumented = np.empty((0, n_hparams + 1))
            # create one hparam config augumented with the corresponding normalized budget for every interim result
            # loop through trials
            for indices, trial_hparams in zip(
                interim_result_indices, hparams_transform
            ):
                # loop through interim results
                for idx in indices:
                    # idx is budget
                    normalized_budget = self.searchspace._normalize_integer(
                        [0, max_budget - 1], idx
                    )
                    augumented_trial_hparams = np.append(
                        deepcopy(trial_hparams), normalized_budget
                    )
                    hparams_augumented = np.vstack(
                        (hparams_augumented, augumented_trial_hparams)
                    )

            # add evaluating trials if impute strategy, i.e. z = [x, max_budget] y = imputed_metric
            if self.include_busy_locations():
                # get params and imputed metrics of evaluating trials
                hparams_busy = self.get_busy_locations(budget=budget)
                imputed_metrics = self.get_imputed_metrics(budget=budget)
                assert (
                    hparams_busy.shape[0] == imputed_metrics.shape[0]
                ), "Number of evaluating trials and imputed metrics needs to be equal, got n_busy_locations: {}, n_imputed_metrics: {}".format(
                    hparams_busy.shape[0], imputed_metrics.shape[0]
                )

                if len(hparams_busy) > 0:
                    # transform hparams
                    hp_trans = np.apply_along_axis(
                        self.searchspace.transform,
                        1,
                        hparams_busy,
                        normalize_categorical=self.normalize_categorical,
                    )
                    # augument with max budget (i.e. always 1 in normalized form)
                    hp_aug = np.append(
                        hp_trans, np.ones(hp_trans.shape[0]).reshape(-1, 1), 1
                    )

                    # append to hparams and metrics
                    hparams_augumented = np.concatenate((hparams_augumented, hp_aug))
                    metrics_flat = np.concatenate((metrics_flat, imputed_metrics))

            X = hparams_augumented
            y = metrics_flat

            assert (
                X.shape[1] == len(self.searchspace.keys()) + 1
            ), "len of Hparam Configs need to be augumented with budget"

        assert X.shape[0] == y.shape[0], "X and y need to same first dim"

        return X, y

    def get_interim_result_idx(self, metric_history, interval=10):
        """helper function for creating hparams with interim results

        get indices of interim results of one trial metric history that will be used for fitting surrogate.

        Note: final metric is always used

        :param metric_history: metric history of one trial
        :param interval: interval of interim metrics to be used. e.g. 10 means every nth metric is used.
        :return: list of indices of the given metric history, that are used for fitting surrogate
        :rtype: list
        """
        max_budget = len(metric_history)
        interim_results_idx = [
            i for i in range(0, max_budget) if (i + 1) % interval == 0
        ]

        # if not enough data points exist for interval use first and final result of metric history
        if not interim_results_idx:
            interim_results_idx = [max_budget - 1]

        # final result is always added
        if interim_results_idx[-1] != (max_budget - 1):
            interim_results_idx.append(max_budget - 1)

        return interim_results_idx

    def include_busy_locations(self):
        """helper function to check if this optimizer is using impute as async strategy

        Returns true when evaluating trials with imputed metrics (liars) should be used for fitting surrogate model

        This is only the case when surrogate is a Gaussian Process with `impute` as async_strategy
        """
        if self.name() == "GP" and self.async_strategy == "impute":
            return True
        else:
            return False
