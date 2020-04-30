import traceback
from copy import deepcopy

import numpy as np

from maggy.optimizer.abstractoptimizer import AbstractOptimizer
from maggy.pruner import Hyperband
from maggy.trial import Trial

from hops import hdfs


# todo which methods should be private todo what about random_state → for reproducability → check skopt for reference
# todo `trial_store` hold all busy trials, think about replacing the `busy_locations` approach with the
#   `trial_store` data
# todo min_delta_x → warn when similar point has been evaluated before → see skopt for reference
# TODO implement resuming trials


class BaseAsyncBO(AbstractOptimizer):
    """Base class for asynchronous bayesian optimization
    # todo explain below
    **async bo**

    **optimization loop details**

    **models**

    saved as dict
        - multifidelity vs. single fidelity

    **pruner**

    todo explain wie min/max direction gehandhabt wird
    todo explain async bo basics

    """

    def __init__(
        self,
        num_warmup_trials=15,
        random_fraction=0.1,
        acq_fun="EI",
        acq_fun_kwargs=None,
        acq_optimizer="lbfgs",
        acq_optimizer_kwargs=None,
        pruner=None,
        pruner_kwargs=None,
    ):
        """
        :param num_warmup_trials: number of random trials at the beginning of experiment
        :type num_warmup_trials: int
        :param random_fraction: fraction of random samples, between [0,1]
        :type random_fraction: float
        :param acq_fun: Function to minimize over the posterior distribution. Can be either
                        - `"LCB"` for lower confidence bound.
                        - `"EI"` for negative expected improvement.
                        - `"PI"` for negative probability of improvement.
        :type acq_fun: str
        :param acq_fun_kwargs: Additional arguments to be passed to the acquisition function.
        :type acq_fun_kwargs: dict
        :param acq_optimizer: Method to minimize the acquisition function. The fitted model
                              is updated with the optimal value obtained by optimizing `acq_func`
                              with `acq_optimizer`.

                              - If set to `"sampling"`, then `acq_func` is optimized by computing
                                  `acq_func` at `n_points` randomly sampled points.
                              - If set to `"lbfgs"`, then `acq_func` is optimized by
                                - Sampling `n_restarts_optimizer` points randomly.
                                - `"lbfgs"` is run for 20 iterations with these points as initial
                                   points to find local minima.
                                - The optimal of these local minima is used to update the prior.
        :param acq_optimizer_kwargs: Additional arguments to be passed to the acquisition optimizer.
        :type acq_optimizer_kwargs: dict
        :param pruner: # todo
        :param pruner_kwargs:

        Attributes
        ----------

        # todo

        """
        super().__init__()

        # from AbstractOptimizer
        # self.final_store # dict of trials
        # selt.trial_store # list of trials → all trials or only unfinished trials ??
        # self.direction
        # self.num_trials
        # self.searchspace

        # configure warmup routine

        self.num_warmup_trials = num_warmup_trials
        self.warmup_sampling = "random"  # todo other options could be latin hypercube
        self.warmup_configs = []  # keeps track of warmup warmup configs

        allowed_sampling_methods = ["random"]
        if self.warmup_sampling not in allowed_sampling_methods:
            raise ValueError(
                "expected warmup_sampling to be in {}, got {}".format(
                    allowed_sampling_methods, self.warmup_sampling
                )
            )

        # configure acquisition function

        allowed_acq_funcs = ["EI"]
        if acq_fun not in allowed_acq_funcs:
            raise ValueError(
                "expected acq_fun to be in {}, got {}".format(
                    allowed_acq_funcs, acq_fun
                )
            )
        self.acq_fun = acq_fun  # calculates the utility for given point and surrogate
        self.acq_func_kwargs = acq_fun_kwargs

        # configure optimizer

        allowed_acq_opt = ["sampling", "lbfgs"]
        if acq_optimizer not in allowed_acq_opt:
            raise ValueError(
                "expected acq_optimizer to be in {}, got {}".format(
                    allowed_acq_opt, acq_optimizer
                )
            )
        self.acq_optimizer = acq_optimizer

        # configure other arguments
        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        self.n_points = acq_optimizer_kwargs.get("n_points", 10000)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get("n_restarts_optimizer", 5)
        self.acq_optimizer_kwargs = acq_optimizer_kwargs

        # configure pruner
        self.pruner = None
        if pruner:
            self.init_pruner(pruner, pruner_kwargs)

        # surrogate model related aruments
        self.busy_locations = (
            []
        )  # each busy location is a dict {"params": hparams_list, "metric": liar} # todo how is budget saved budget
        self.base_model = None  # estimator that has not been fit on any data.
        # todo explain how models are saved → dict, 0 is single fidelity
        self.models = {}  # fitted model of the estimator
        self.random_fraction = random_fraction

        # configure logger

        self.log_file = "hdfs:///Projects/demo_deep_learning_admin000/Logs/asyncbo_{}.log".format(
            self.name()
        )  # todo make dynamic
        if not hdfs.exists(self.log_file):
            hdfs.dump("", self.log_file)
        self.fd = hdfs.open_file(self.log_file, flags="w")
        self._log("Initialized Logger")

    def initialize(self):
        self.warmup_routine()
        self.init_model()

    def get_suggestion(self, trial=None):
        try:
            if trial:
                self._log("Last finished Trial: {}".format(trial.trial_id))
            else:
                self._log("no previous finished trial")

            # remove hparams of last finished trial from `busy_locations`
            if trial:
                self._cleanup_busy_locations(trial)

            # check if experiment has finished
            if self._experiment_finished():
                return None

            # pruning routine
            if self.pruner:
                next_trial_info = self.pruner.pruning_routine()
                if next_trial_info == "IDLE":
                    self._log(
                        "Worker is IDLE and has to wait until a new trial can be scheduled \n"
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
                        self.get_hparams_dict(trial_ids=parent_trial_id)[
                            parent_trial_id
                        ]
                    )
                    next_trial = self.create_trial(
                        hparams=parent_trial_hparams, budget=next_trial_info["budget"]
                    )
                    # report new trial id to pruner
                    self.pruner.report_trial(
                        original_trial_id=parent_trial_id,
                        new_trial_id=next_trial.trial_id,
                    )
                    self._log(
                        "Use hparams from promoted Trial {}. \n Start Trial {}: {} \n".format(
                            parent_trial_id, next_trial.trial_id, next_trial.params
                        )
                    )
                    return next_trial
                else:
                    # start sampling procedure with given budget
                    budget = next_trial_info["budget"]
            else:
                budget = 0

            # check if there are still trials in the warmup buffer
            if self.warmup_configs:
                self._log("take sample from warmup buffer")
                next_trial_params = self.warmup_configs.pop()
                next_trial = self.create_trial(hparams=next_trial_params, budget=budget)
                # report new trial id to pruner
                if self.pruner:
                    self.pruner.report_trial(
                        original_trial_id=None, new_trial_id=next_trial.trial_id
                    )
                # todo evtl auch erst unten returnen und somit report call sparen
                self._log(
                    "Start Trial {}: {} \n".format(
                        next_trial.trial_id, next_trial.params
                    )
                )
                return next_trial

            # update model with latest observations
            # skip model building if we already have a bigger model
            if max(list(self.models.keys()) + [-np.inf]) <= budget:
                self.update_model(budget)

            # in case there is no model yet or random fraction applies, sample randomly
            if not self.models or np.random.rand() < self.random_fraction:
                hparams = self.searchspace.get_random_parameter_values(1)[0]
                self._log("sampled randomly: {}".format(hparams))
            else:
                # sample from largest model
                max_budget = max(self.models.keys())
                hparams = self.sampling_routine(max_budget)
                self._log(
                    "sampled from model with budget {}: {}".format(max_budget, hparams)
                )

            # create Trial object
            next_trial = self.create_trial(hparams, budget=budget)
            # report new trial id to pruner
            if self.pruner:
                self.pruner.report_trial(
                    original_trial_id=None, new_trial_id=next_trial.trial_id
                )
            self._log(
                "Start Trial {}: {} \n".format(next_trial.trial_id, next_trial.params)
            )
            return next_trial

        except BaseException:
            self._log(traceback.format_exc())
            self._close_log()

            if self.pruner:
                self.pruner._close_log()

    def finalize_experiment(self, trials):
        self._log("Experiment finished")
        self._close_log()

        # todo eliminiate
        if self.pruner:
            self.pruner._close_log()

        return

    def init_model(self):
        """initializes the surrogate model of the gaussian process

        the model gets created with the right parameters, but is not fit with any data yet. the `base_model` will be
        cloned in `update_model` and fit with observation data
        """
        raise NotImplementedError

    def init_pruner(self, pruner, pruner_kwargs):
        """intializes pruner

        :param pruner: name of pruner. so far only "hyperband" implemented
        :type pruner: str
        :param pruner_kwargs: dict of pruner kwargs
        :type pruner_kwargs: dict
        :return: initiated pruner instance
        """
        allowed_pruners = ["hyperband"]
        if pruner not in allowed_pruners:
            raise ValueError(
                "expected pruner to be in {}, got {}".format(allowed_pruners, pruner)
            )

        if pruner == "hyperband":
            self.pruner = Hyperband(
                trial_metric_getter=self.get_metrics_dict, **pruner_kwargs
            )

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

        todo add other options s.a. latin hypercube
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

        self._log("warmup configs: {}".format(self.warmup_configs))

    # todo, evtl. outsource to helpers or abstract optimizer or trial. possibly obsolete when trial has param budget
    def create_trial(self, hparams, budget=0):
        """helper function to create trial with budget

        `budget == 0` means that it is a single fidelity optimization and budget does not need to be passed to Trial
        and hence training function

        :param hparams: hparam dict
        :type hparams: dict
        :param budget: budget for trial. implemented first version of Hyperband
        :type budget: int
        :return: Trial object with specified params
        :rtype: Trial
        """
        if budget > 0:
            hparams["budget"] = budget
        return Trial(hparams, trial_type="optimization")

    def get_trial(self, trial_ids):
        """return Trial or list of Trials with `trial_id` from `final_store`

        # todo, probably eliminate

        :param trial_ids: single trial id or list of trial ids of the requested trials
        :type trial_ids: str|list[str]
        :return: Trial/ Trials with specified id
        :rtype: Trial|list[Trial]
        """
        if isinstance(trial_ids, str):
            # return single trial object
            trial = [trial for trial in self.final_store if trial.trial_id == trial_ids]
            if len(trial) > 0:
                return trial[0]
            else:
                self._log(
                    "There is no trial with id {} in final_store".format(trial_ids)
                )
        else:
            # return list of trials, `trial_id` is list of `trial_id`
            trials = [
                trial for trial in self.final_store if trial.trial_id in trial_ids
            ]
            return trials

    def _acquisition_function(self):
        """calculates the utility for given point and surrogate"""

    def _maximize_acq_function(self):
        """maximizes acquisition function"""

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

    def _cleanup_busy_locations(self, trial):
        """deletes hparams of `trial` from `busy_locations`

        .. note::  alternatively we could compare hparams of all finished trials with busy locations, would take longer
                  to compute but at the same would ensure consistency → ask Moritz

        :param trial: finished Trial
        :type trial: Trial
        """
        # convert to list, because `busy_locations` stores params in list format
        hparams = self.searchspace.dict_to_list(trial.params)

        # find and delete from busy location
        index = next(
            (
                index
                for (index, d) in enumerate(self.busy_locations)
                if d["params"] == hparams
            ),
            None,
        )
        try:
            del self.busy_locations[index]
            # self._log("{} was deleted from busy_locations".format(hparams))
        except TypeError:
            pass
            # self._log("{} was not in busy_locations".format(hparams))

    def get_hparams_array(self, include_busy_locations=False, budget=0):
        """returns array of already evaluated hparams + optionally hparams that are currently evaluated

        The order of the returned hparams is the same as in `final_store`

        :param include_busy_locations: If True, add currently evaluating hparam configs
        :type include_busy_locations: bool
        :param budget: If not None, only trials with this budget are returned
        :type budget: None|int
        :return: array of hparams, shape (n_finished_hparam_configs, n_hparams)
        :rtype: np.ndarray[np.ndarray]
        """
        include_trial = lambda x: x == budget  # noqa: E731

        hparams = np.array(
            [
                self.searchspace.dict_to_list(trial.params)
                for trial in self.final_store
                # include trials with given budget or include all trials if no budget is given
                if budget == 0
                or budget is None
                or include_trial(trial.params["budget"])
            ]
        )

        if include_busy_locations and len(self.busy_locations):
            # todo if/else wont be necessary when budget is attr of Trial
            if budget == 0 or budget is None:
                hparams_busy = np.array(
                    [location["params"] for location in self.busy_locations]
                )
            else:
                hparams_busy = np.array(
                    [
                        np.append(location["params"], budget)
                        for location in self.busy_locations
                        if location["budget"] == budget
                    ]
                )
            hparams = np.concatenate((hparams, hparams_busy))

        return hparams

    def get_hparams_dict(self, trial_ids="all"):
        """returns dict of hparams of finished trials with `trial_id` as key and hparams dict as value

        :param trial_ids: trial_id or list of trial_ids that should be returned.
                          If set to default ("all"), return all trials
        :type trial_ids: list[str]|str
        :return: dict of trial_ids and hparams. Example: {`trial_id1`: `hparam_dict1`, ... , `trial_idn`: `hparam_dictn`}
        :rtype: dict
        """
        # return trials with specified trial_ids or return all trials
        include_trial = (
            lambda x: x == trial_ids or x in trial_ids or trial_ids == "all"
        )  # noqa: E731

        hparam_dict = {
            trial.trial_id: trial.params
            for trial in self.final_store
            if include_trial(trial.trial_id)
        }

        return hparam_dict

    def get_metrics_array(self, include_busy_locations=False, budget=0):
        """returns array of final metrics + optionally imputed metrics of currently evaluating trials

        The order of the returned metrics is the same as in `final_store`

        In case that the optimization `direction` is `max`, negate the metrics so it becomes a `min` problem

        :param include_busy_locations: If True, add imputed metrics of currently evaluating trials
        :type include_busy_locations: bool
        :param budget: If not None, only trials with this budget are returned
        :type budget: None|int
        :return: array of hparams, shape (n_final_metrics,)
        :rtype: np.ndarray[float]
        """
        include_trial = lambda x: x == budget  # noqa: E731

        metrics = np.array(
            [
                trial.final_metric
                for trial in self.final_store
                # include trials with given budget or include all trials if no budget is given
                if budget == 0
                or budget is None
                or include_trial(trial.params["budget"])
            ]
        )

        if include_busy_locations and len(self.busy_locations):
            metrics_busy = np.array(
                [
                    location["metric"]
                    for location in self.busy_locations
                    if budget == 0 or budget is None or location["budget"] == budget
                ]
            )
            metrics = np.concatenate((metrics, metrics_busy))

        if self.direction == "max":
            metrics = -metrics

        return metrics

    def get_metrics_dict(self, trial_ids="all"):
        """returns dict of final metrics with `trial_id` as key and final metric as value

        In case that the optimization `direction` is `max`, negate the metrics so it becomes a `min` problem

        :param trial_ids: trial_id or list of trial_ids that should be returned.
                          If set to default ("all"), return all trials
        :type trial_ids: list[str]|str
        :return: dict of trial_ids and final_metrics. Example: {`trial_id1`: `metric1`, ... , `trial_idn`: `metricn`}
        :rtype: dict
        """
        if self.direction == "max":
            metric_multiplier = -1
        else:
            metric_multiplier = 1

        # return trials with specified trial_ids or return all trials
        include_trial = (
            lambda x: x == trial_ids or x in trial_ids or trial_ids == "all"
        )  # noqa: E731

        metrics = {
            trial.trial_id: trial.final_metric * metric_multiplier
            for trial in self.final_store
            if include_trial(trial.trial_id)
        }

        return metrics

    def ybest(self, budget=0):
        """Returns best metric of all currently finalized trials

        Maximization problems are converted to minimization problems
        I.e. if the optimization direction is `max`, returns the negated max value

        :param budget: the budget for which ybest should be calculated
        :type budget: int
        :return: worst metric of all currently finalized trials
        :rtype: float
        """
        metric_history = self.get_metrics_array(budget=budget)
        return np.min(metric_history)

    def yworst(self, budget=0):
        """Returns worst metric of all currently finalized trials

        Maximization problems are converted to minimization problems
        I.e. if the optimization direction is `max`, returns the negated min value

        :param budget: the budget for which yworst should be calculated
        :type budget: int
        :return: best metric of all currently finalized trials
        :rtype: float
        """
        metric_history = self.get_metrics_array(budget=budget)
        return np.max(metric_history)

    def ymean(self, budget=0):
        """Returns best metric of all currently finalized trials

        Maximization problems are converted to minimization problems
        I.e. if the optimization direction is `max`, returns the mean of negated metrics

        :param budget: the budget for which ymean should be calculated
        :type budget: int
        :return: mean of all currently finalized trials metrics
        :rtype: float
        """
        metric_history = self.get_metrics_array(budget=budget)
        return np.mean(metric_history)

    def _log(self, msg):
        if not self.fd.closed:
            self.fd.write((msg + "\n").encode())

    def _close_log(self):
        if not self.fd.closed:
            self.fd.flush()
            self.fd.close()
