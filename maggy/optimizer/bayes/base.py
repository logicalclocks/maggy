import traceback
import time
from copy import deepcopy

import numpy as np

from maggy.optimizer.abstractoptimizer import AbstractOptimizer
from maggy.pruner import Hyperband
from maggy.trial import Trial

from hops import hdfs


# todo which methods should be private
# todo what about random_state → for reproducability → check skopt for reference
# todo min_delta_x → warn when similar point has been evaluated before → see skopt for reference
# TODO poss. shift `include_busy_locations` logic to gp.py if it is only used ther
# TODO implement resuming trials
# TODO when intermediate trial metric per budget is implemented, update models of lower budgets with intermediate
#  results of trials with larger budget
# TODO check that hparams have valid type,


class BaseAsyncBO(AbstractOptimizer):
    """Base class for asynchronous bayesian optimization

    Do not initialize this class! only it's subclasses

    **async bo**

    todo explain async bo basics

    **optimization loop details**


    **optimization direction**

    all optimizations are converted to minimizations problems via the `get_metrics_dict()` and `get_metrics_array()`
    methods, where the metrics are negated in case of a maximization.
    In the `final_store` the original metric values are saved.

    **models**

    The surrogate models for different budgets are saved in the `models` dict with the budgets as key. In case of
    a single fidelity optimization without a pruner. The only model has the key `0`.
    Sample new hparam configs always from the largest model available (i.e. biggest budget) or in case of async bo with
    imputing strategy + multi-fidelity pruner, sample from the model that has same budget as the trial

    **imputing busy trials** (poss. shift to simple.py)

    the imputed metrics are calculated on the fly in `get_metrics_array(include_busy_locations=True, budget=`budget`)`
    for currently evaluating trials that were sampled from model with `budget`

    **pruner**

    Pruners can be run as a subroutine of an optimizer. It's interface `pruner.pruning_routine()` is called at the beginning
    of the `get_suggestion()` method and returns the hparam config and budget for the trial. The `trial_id` of the
    newly created Trial object is reported back to the pruner via `pruner.report_trial()`.
    """

    def __init__(
        self,
        num_warmup_trials=15,
        random_fraction=0.1,
        pruner=None,
        pruner_kwargs=None,
    ):
        """
        :param num_warmup_trials: number of random trials at the beginning of experiment
        :type num_warmup_trials: int
        :param random_fraction: fraction of random samples, between [0,1]
        :type random_fraction: float
        :param pruner: # todo
        :param pruner_kwargs:

        Attributes
        ----------

        max_model (bool): If True, always sample from the largest model available in multi fidelity optimization.
                          Else, sample from the model that has same budget as the trial.
                          False, if async bo algorithm with a async_strategy == `impute` and bandit based pruner is used
        models (dict): The surrogate models for different budgets are saved in the `models` dict with the budgets as key.
                       In case of a single fidelity optimization without a pruner. The only model has the key `0`.
        warm_up_configs(list[dict]): list of hparam configs used for warming up
        sampling_time_start (float): helper variable to calculate time needed for calculating next suggestion, i.e
                                       when sampling from model (`sample_type`=="model"). Calulating the time happens
                                       in `create_trial()`

        """
        super().__init__()

        # from AbstractOptimizer
        # self.final_store # dict of trials
        # selt.trial_store # list of trials → all trials or only unfinished trials ??
        # self.direction
        # self.num_trials
        # self.searchspace

        # configure pruner
        self.pruner = None
        if pruner:
            self.init_pruner(pruner, pruner_kwargs)

        # configure number of trials ( for maggy progress bar )
        if pruner:
            self.num_trials = (
                self.pruner.num_trials()
            )  # todo does not work yet, progress bar uses num_trials from kwarg of optimizer

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
        self.max_model = True

        # configure logger
        self.log_file = "hdfs:///Projects/{}/Logs/optimizer_{}_{}.log".format(
            hdfs.project_name(), self.name(), self.pruner.name() if self.pruner else ""
        )
        if not hdfs.exists(self.log_file):
            hdfs.dump("", self.log_file)
        self.fd = hdfs.open_file(self.log_file, flags="w")
        self._log("Initialized Logger")
        self._log("Initilized Optimizer {}: \n {}".format(self.name(), self.__dict__))

        # helper variable to calculate time needed for calculating next suggestion
        self.sampling_time_start = 0.0

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
        try:
            self.sampling_time_start = time.time()
            if trial:
                self._log(
                    "Last finished Trial: {} with params {}".format(
                        trial.trial_id, trial.params
                    )
                )
            else:
                self._log("no previous finished trial")

            # todo eliminate
            self._log("Trial Store:")
            for key, val in self.trial_store.items():
                self._log("{}: {} \n".format(key, val.params))

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
                next_trial = self.create_trial(
                    hparams=next_trial_params, sample_type="random", run_budget=budget
                )
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
            if self.max_model:
                # skip model building if we already have a bigger model
                if max(list(self.models.keys()) + [-np.inf]) <= budget:
                    self.update_model(budget)
            else:
                self.update_model(budget)

            # in case there is no model yet or random fraction applies, sample randomly
            if not self.models or np.random.rand() < self.random_fraction:
                hparams = self.searchspace.get_random_parameter_values(1)[0]
                next_trial = self.create_trial(
                    hparams=hparams, sample_type="random", run_budget=budget
                )
                self._log("sampled randomly: {}".format(hparams))
            else:
                if self.max_model:
                    # sample from largest model
                    model_budget = max(self.models.keys())
                else:
                    # sample from model with same budget as trial
                    model_budget = budget
                hparams = self.sampling_routine(model_budget)
                next_trial = self.create_trial(
                    hparams=hparams,
                    sample_type="model",
                    run_budget=budget,
                    model_budget=model_budget,
                )
                self._log(
                    "sampled from model with budget {}: {}".format(
                        model_budget, hparams
                    )
                )

            # check if Trial with same hparams has already been created
            if self.hparams_exist(trial=next_trial):
                self._log("Sample randomly to encourage exploration")
                hparams = self.searchspace.get_random_parameter_values(1)[0]
                next_trial = self.create_trial(
                    hparams=hparams, sample_type="random", run_budget=budget
                )

            # report new trial id to pruner
            if self.pruner:
                self.pruner.report_trial(
                    original_trial_id=None, new_trial_id=next_trial.trial_id
                )
            self._log(
                "Start Trial {}: {}, {} \n".format(
                    next_trial.trial_id, next_trial.params, next_trial.info_dict
                )
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
    def create_trial(self, hparams, sample_type, run_budget=0, model_budget=None):
        """helper function to create trial with budget and trial_dict

        `run_budget == 0` means that it is a single fidelity optimization and budget does not need to be passed to Trial
        and hence training function

        :param hparams: hparam dict
        :type hparams: dict
        :param sample_type: specifies how the hapram config was sampled. can be "random", "promoted", "model"
        :type sample_type: str
        :param run_budget: budget for trial or 0 if there is no budget, i.e. single fidelity optimization
        :type run_budget: int
        :param model_budget: If `sampled` == `model`, specifies from which model the sample was generated
        :type model_budget: int
        :return: Trial object with specified params
        :rtype: Trial
        """
        # validations
        allowed_sample_type_values = ["random", "model", "promoted"]
        if sample_type not in allowed_sample_type_values:
            raise ValueError(
                "expected sample_type to be in {}, got {}".format(
                    allowed_sample_type_values, sample_type
                )
            )
        if sample_type == "model" and model_budget is None:
            raise ValueError(
                "expected `model_budget` because sample_type==`model`, got None"
            )

        # calculate time needed for sampling
        sampling_time = time.time() - self.sampling_time_start
        self.sampling_time_start = 0.0

        # init trial info dict
        trial_info_dict = {
            "run_budget": run_budget,
            "sample_type": sample_type,
            "sampling_time": sampling_time,
        }
        if model_budget is not None:
            trial_info_dict["model_budget"] = model_budget

        # todo legacy → in the long run have budget as explicit attr of trial object
        if run_budget > 0:
            hparams["budget"] = run_budget

        return Trial(hparams, trial_type="optimization", info_dict=trial_info_dict)

    def hparams_exist(self, trial):
        """Checks if Trial with hparams and budget has already been started

        :param trial: trial instance to validate
        :type trial: Trial
        :return: True, if trial with same params already exists
        :rtype: bool

        """
        # check in finished trials
        # todo when budget becomes attr of Trial object ( and not part of params anymore ), adapt the comparison
        for idx, finished_trial in enumerate(self.final_store):
            if trial.params == finished_trial.params:
                self._log(
                    "WARNING: Hparams {} are equal to params of finished trial no. {}: {}".format(
                        trial.params, idx, finished_trial.to_dict()
                    )
                )
                return True

        # check in currently evaluating trials
        for trial_id, busy_trial in self.trial_store.items():
            if trial.params == busy_trial.params:
                self._log(
                    "WARNING: Hparams {} are equal to currently evaluating Trial: {}".format(
                        trial.params, busy_trial.to_dict()
                    )
                )
                return True

        return False

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

    def get_hparams_array(self, include_busy_locations=False, budget=0):
        """returns array of hparams that were evaluated with `budget`
        + optionally currently evaluating hparams that were sampled from the model with `budget`

        The order of the returned hparams is the same as in `final_store`

        :param include_busy_locations: If True, add currently evaluating hparam configs
        :type include_busy_locations: bool
        :param budget: budget of trials to return
        :type budget: int
        :return: array of hparams, shape (n_finished_hparam_configs, n_hparams)
        :rtype: np.ndarray[np.ndarray]

        # todo when budget becomes attr of Trial object ( and not part of params anymore ) adapt
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

        if include_busy_locations:
            # validate that optimizer is useing correct async_strategy
            if not (self.name() == "GP" and self.async_strategy == "impute"):
                raise ValueError(
                    "Optimizer GP wants to include busy locations, expected async_strategy == `impute`. Got {}".format(
                        self.async_strategy
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

            if len(hparams_busy) > 0:
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
        """returns array of final metrics of trials that were run with `budget`
         + optionally imputed metrics of currently evaluating trials that were sampled from model with `budget`

        The order of the returned metrics is the same as in `final_store`

        In case that the optimization `direction` is `max`, negate the metrics so it becomes a `min` problem

        :param include_busy_locations: If True, add imputed metrics of currently evaluating trials
        :type include_busy_locations: bool
        :param budget: budget of trials to return
        :type budget: int
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

        if include_busy_locations:
            # validate that optimizer is useing correct async_strategy
            if not (self.name() == "GP" and self.async_strategy == "impute"):
                raise ValueError(
                    "Optimizer GP wants to include busy locations, expected async_strategy == `impute`. Got {}".format(
                        self.async_strategy
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

            if len(metrics_busy) > 0:
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
