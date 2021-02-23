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
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np

from maggy.core.environment.singleton import EnvSing
from maggy.pruner import Hyperband
from maggy.trial import Trial


class AbstractOptimizer(ABC):
    def __init__(self, pruner=None, pruner_kwargs=None):
        """
        :param pruner: name of pruning algorithm to use. So far only `hyperband` supported
        :type pruner: str
        :param pruner_kwargs: dict of arguments for initializing pruner. See pruner classes for reference.
        :type pruner_kwargs: dict
        """
        self.searchspace = None
        self.num_trials = None
        self.trial_store = None
        self.final_store = None
        self.direction = None
        self.pruner = None

        # configure pruner
        if pruner:
            self.init_pruner(pruner, pruner_kwargs)

        # logger variables
        self.log_file = None
        self.fd = None

        # helper variable to calculate time needed for calculating next suggestion
        self.sampling_time_start = 0.0

    @abstractmethod
    def initialize(self):
        """
        A hook for the developer to initialize the optimizer.
        """
        pass

    @abstractmethod
    def get_suggestion(self, trial=None):
        """
        Return a `Trial` to be assigned to an executor, or `None` if there are
        no trials remaining in the experiment.

        :param trial: last finished trial by an executor

        :rtype: Trial or None
        """
        pass

    @abstractmethod
    def finalize_experiment(self, trials):
        """
        This method will be called before finishing the experiment. Developers
        can implement this method e.g. for cleanup or extra logging.
        """
        pass

    def name(self):
        return str(self.__class__.__name__)

    def _initialize_logger(self, exp_dir):
        """Initialize logger of optimizer

        :param exp_dir: path of experiment directory
        :rtype exp_dir: str
        """
        env = EnvSing.get_instance()
        # configure logger
        self.log_file = exp_dir + "/optimizer.log"
        if not env.exists(self.log_file):
            env.dump("", self.log_file)
        self.fd = env.open_file(self.log_file, flags="w")
        self._log("Initialized Optimizer Logger")

    def _initialize(self, exp_dir):
        """
        initialize the optimizer and configure logger.

        :param exp_dir: path of experiment directory
        :rtype exp_dir: str
        """
        # init logger of optimizer
        self._initialize_logger(exp_dir=exp_dir)
        # optimizer intitialization routine
        self.initialize()
        self._log("Initilized Optimizer {}: \n {}".format(self.name(), self.__dict__))
        # init logger of pruner
        if self.pruner:
            self.pruner.initialize_logger(exp_dir=exp_dir)

    def _finalize_experiment(self, trials):
        # run optimizer specific finalize routine
        self.finalize_experiment(trials)

        self._log("Experiment finished")
        self._close_log()

        if self.pruner:
            self.pruner._close_log()

        return

    def _log(self, msg):
        if self.fd and not self.fd.closed:
            msg = datetime.now().isoformat() + ": " + str(msg)
            self.fd.write(EnvSing.get_instance().str_or_byte(msg + "\n"))

    def _close_log(self):
        if not self.fd.closed:
            self.fd.flush()
            self.fd.close()

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

    def hparams_exist(self, trial):
        """Checks if Trial with hparams and budget has already been started

        :param trial: trial instance to validate
        :type trial: Trial
        :return: True, if trial with same params already exists
        :rtype: bool

        """

        def remove_budget(trial_params):
            """In multi fidelity setting budget is added as key to trial.params.
            Needs to be removed here to compare actual params defined in search space
            """
            return dict(
                (k, trial_params[k])
                for k in self.searchspace.keys()
                if k in trial_params
            )

        # check in finished trials
        # todo when budget becomes attr of Trial object ( and not part of params anymore ), adapt the comparison
        for idx, finished_trial in enumerate(self.final_store):
            if remove_budget(trial.params) == remove_budget(finished_trial.params):
                self._log(
                    "WARNING Duplicate Config: Hparams {} are equal to params of finished trial no. {}: {}".format(
                        trial.params, idx, finished_trial.trial_id
                    )
                )
                return True

        # check in currently evaluating trials
        for trial_id, busy_trial in self.trial_store.items():
            if remove_budget(trial.params) == remove_budget(busy_trial.params):
                self._log(
                    "WARNING Duplicate Config: Hparams {} are equal to currently evaluating Trial: {}".format(
                        trial.params, busy_trial.trial_id
                    )
                )
                return True

        return False

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

    def create_trial(self, hparams, sample_type, run_budget=0, model_budget=None):
        """helper function to create trial with budget and trial_dict

        `run_budget == 0` means that it is a single fidelity optimization and budget does not need to be passed to Trial
        and hence training function

        :param hparams: hparam dict
        :type hparams: dict
        :param sample_type: specifies how the hapram config was sampled.
                            can take values:
                                - "random": config was sampled randomly
                                - "random_forced": config was sampled randomly because model sampling returned a config that
                                             already exists
                                - "promoted": config was promoted in multi fidelity bandit based setting
                                - "model": config was sampled from surrogate by optimizing acquisiton function
        :type sample_type: str
        :param run_budget: budget for trial or 0 if there is no budget, i.e. single fidelity optimization
        :type run_budget: int
        :param model_budget: If sample_type == `model`, specifies from which model the sample was generated
        :type model_budget: int
        :return: Trial object with specified params
        :rtype: Trial
        """
        # validations
        allowed_sample_type_values = [
            "random",
            "random_forced",
            "model",
            "promoted",
            "grid",
        ]
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
