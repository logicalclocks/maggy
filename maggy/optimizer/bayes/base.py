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
        random_fraction=0.1,
        interim_results=False,
        interim_results_interval=10,
        pruner=None,
        pruner_kwargs=None,
    ):
        """
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
        normalize_categorical (bool): If True, the encoded categorical hparam is also max-min
                                      normalized between 0 and 1 in searchspace.transform()
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
        self.interim_results = (
            interim_results  # todo do I maybe need to init this before init pruner
        )
        self.interim_results_interval = interim_results_interval
        self.max_model = True

        # configure logger
        self.log_file = "hdfs:///Projects/{}/Experiments_Logs/optimizer_{}_{}.log".format(
            hdfs.project_name(), self.name(), self.pruner.name() if self.pruner else ""
        )
        if not hdfs.exists(self.log_file):
            hdfs.dump("", self.log_file)
        self.fd = hdfs.open_file(self.log_file, flags="w")
        self._log("Initialized Logger")
        self._log("Initilized Optimizer {}: \n {}".format(self.name(), self.__dict__))

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

            if self.interim_results:
                # there is only one model. i.e. also in multi fidelity setting
                model_budget = 0
            elif self.max_model:
                # always sample from largest model avaliable. i.e. in multi fidelity setting
                model_budget = max(list(self.models.keys()) + [budget])
            else:
                # sample from model with same budget as run budget of trial
                model_budget = budget

            # update model
            if self.max_model:
                # skip model building if we already have a bigger model
                if max(list(self.models.keys()) + [-np.inf]) <= budget:
                    self.update_model(model_budget)
            else:
                self.update_model(model_budget)

            if not self.models or np.random.rand() < self.random_fraction:
                # in case there is no model yet or random fraction applies, sample randomly
                hparams = self.searchspace.get_random_parameter_values(1)[0]
                next_trial = self.create_trial(
                    hparams=hparams, sample_type="random", run_budget=budget
                )
                self._log("sampled randomly: {}".format(hparams))
            else:
                # sample from model with model budget
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
        :param model_budget: If sample_type == `model`, specifies from which model the sample was generated
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

    def get_hparams_array(self, budget=0):
        """returns array of hparams that were evaluated with `budget`

        The order of the returned hparams is the same as in `final_store`

        :param budget: budget of trials to return
        :type budget: int
        :return: array of hparams, shape (n_finalized_trials, n_hparams)
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

    def get_metrics_array(self, budget=0, interim_metrics=False):
        """returns final metrics or metric histories of trials that were run with `budget`

        The order of the returned metrics is the same as in `final_store`

        In case that the optimization `direction` is `max`, negate the metrics so it becomes a `min` problem

        :param budget: budget of trials to return
        :type budget: int
        :param interim_metrics: If true return metric histories, Else return final metrics
        :type interim_metrics: bool
        :return: array of final metrics; shape (n_finalized_trials,) or
                 array of arrays of metric histories; shape (n_finalized_trials, max budget) if all trials were trained
                 on max budget, else (n_finalized_trials,) → ragged array
        :rtype: np.ndarray[float|np.ndarray]
        """

        include_trial = lambda x: x == budget  # noqa: E731

        metrics = []
        for trial in self.final_store:
            # include trials with given budget or include all trials if no budget is given
            if budget == 0 or budget is None or include_trial(trial.params["budget"]):
                if interim_metrics:
                    # append whole metric history of trial, note the conversion to np.array
                    m = np.array(trial.metric_history)
                else:
                    # append final metric of trial
                    m = trial.final_metric

                metrics.append(m)

        metrics = np.array(metrics)

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

        Note: first and final metric are always used

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
            interim_results_idx = [0, max_budget - 1]

        # first result is always added
        if interim_results_idx[0] != 0:
            interim_results_idx.insert(0, 0)

        # final result is always added
        if interim_results_idx[-1] != (max_budget - 1):
            interim_results_idx.append(max_budget - 1)

        return interim_results_idx

    def get_max_budget(self):
        """returns maxmimum budget of experiment

        i.e. maximum of all trials in the experiment

        If optimizer uses a pruner, max budget can be retrieved directly from pruner,
        Else use metric history of first finalized trial. It will be always trained with max budget in single fidelity
        setting

        :return: maxmimum budget of experiment
        :rtype: int
        """

        if self.pruner:
            return self.pruner.max_budget
        else:
            if len(self.final_store) == 0:
                raise ValueError(
                    "At least one finalized Trial is necessary to calculate max budget"
                )

            # first finalized trial is always evaluated on max budget
            return len(self.final_store[0].metric_history)

    def include_busy_locations(self):
        """helper function to check if this optimizer is using impute as async strategy

        Returns true when evaluating trials with imputed metrics (liars) should be used for fitting surrogate model

        This is only the case when surrogate is a Gaussian Process with `impute` as async_strategy
        """
        if self.name() == "GP" and self.async_strategy == "impute":
            return True
        else:
            return False

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
