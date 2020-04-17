import traceback

import numpy as np

from maggy.optimizer.abstractoptimizer import AbstractOptimizer
from maggy.trial import Trial

from hops import hdfs


# todo which methods should be private


class BaseAsyncBO(AbstractOptimizer):
    """Base class for asynchronous bayesian optimization

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
        self.warmup_trial_buffer = []  # keeps track of warmup trials

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

        # record other arguments
        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        self.n_points = acq_optimizer_kwargs.get("n_points", 10000)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get("n_restarts_optimizer", 5)
        self.acq_optimizer_kwargs = acq_optimizer_kwargs

        # todo configure pruner

        self.pruner = (
            pruner  # class vs. instance vs. string ?? → same discussion for acq_fun
        )
        self.pruner_kwargs = pruner_kwargs

        # surrogate model related aruments

        self.busy_locations = (
            []
        )  # each busy location is a dict {"params": hparams_list, "metric": liar}
        self.base_model = None  # estimator that has not been fit on any data.
        self.model = None  # fitted model of the estimator
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

            # check if there are still trials in the warmup buffer
            if self.warmup_trial_buffer:
                self._log("take sample from warmup buffer")
                return self.warmup_trial_buffer.pop()

            # update model with latest observations
            self.update_model()

            # in case there is no model yet or random fraction applies, sample randomly
            # todo in case of BOHB/ASHA model is a dict, maybe it should be dict for every case
            if not self.model or np.random.rand() < self.random_fraction:
                hparams = self.searchspace.get_random_parameter_values(1)[0]
                self._log("sampled randomly: {}".format(hparams))
                return Trial(hparams)

            # sample best hparam config from model
            hparams = self.sampling_routine()

            return Trial(hparams)

        except BaseException:
            self._log(traceback.format_exc())
            self.fd.flush()
            self.fd.close()

    def finalize_experiment(self, trials):
        # close logfile
        self.fd.flush()
        self.fd.close()

        return

    def init_model(self):
        """initializes the surrogate model of the gaussian process

        the model gets created with the right parameters, but is not fit with any data yet. the `base_model` will be
        cloned in `update_model` and fit with observation data
        """
        raise NotImplementedError

    def update_model(self):
        """update surrogate model with new observations

        Use observations of finished trials + liars from busy trials to build model.
        Only build model when there are at least as many observations as hyperparameters
        """
        raise NotImplementedError

    def sampling_routine(self):
        """Samples new config from surrogate model

        This methods holds logic for:

        - maximizing acquisition function based on current model and observations
        - async logic: i.e. imputing busy_locations with a liar to encourage diversity in sampling

        :return: hyperparameter config that minimizes the acquisition function
        :rtype: dict
        """
        raise NotImplementedError

    def warmup_routine(self):
        """implements logic for warming up bayesian optimization through random sampling by adding hparam configs to
        `warmup_trial_buffer`

        todo add other options s.a. latin hypercube
        """

        # generate warmup hparam configs
        if self.warmup_sampling == "random":
            warmup_configs = self.searchspace.get_random_parameter_values(
                self.num_warmup_trials
            )
        else:
            raise NotImplementedError(
                "warmup sampling {} doesnt exist, use random".format(
                    self.warmup_sampling
                )
            )

        self._log("warmup configs: {}".format(warmup_configs))

        # add configs to trial buffer
        for hparams in warmup_configs:
            self.warmup_trial_buffer.append(Trial(hparams, trial_type="optimization"))

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

        if len(self.final_store) >= self.num_trials:
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
            self._log("{} was deleted from busy_locations".format(hparams))
        except TypeError:
            self._log("{} was not in busy_locations".format(hparams))

    def get_hparams(self, include_busy_locations=False):
        """returns array of already evaluated hparams + optionally hparams that are currently evaluated

        :param include_busy_locations: If True, add currently evaluating hparam configs
        :type include_busy_locations: bool
        :return: array of hparams, shape (n_finished_hparam_configs, n_hparams)
        :rtype: np.ndarray[np.ndarray]
        """
        hparams = np.array(
            [self.searchspace.dict_to_list(trial.params) for trial in self.final_store]
        )

        if include_busy_locations and len(self.busy_locations):
            hparams_busy = np.array(
                [location["params"] for location in self.busy_locations]
            )
            hparams = np.concatenate((hparams, hparams_busy))

        return hparams

    def get_metrics(self, include_busy_locations=False):
        """returns array of final metrics + optionally imputed metrics of currently evaluating trials

        In case that the optimization `direction` is `max`, negate the metrics so it becomes a `min` problem

        :param include_busy_locations: If True, add imputed metrics of currently evaluating trials
        :type include_busy_locations: bool
        :return: array of hparams, shape (n_final_metrics,)
        :rtype: np.ndarray[float]
        """
        metrics = np.array([trial.final_metric for trial in self.final_store])

        if include_busy_locations and len(self.busy_locations):
            metrics_busy = np.array(
                [location["metric"] for location in self.busy_locations]
            )
            metrics = np.concatenate((metrics, metrics_busy))

        if self.direction == "max":
            metrics = -metrics

        return metrics

    def ybest(self):
        """Returns best metric of all currently finalized trials

        Maximization problems are converted to minimization problems
        I.e. if the optimization direction is `max`, returns the negated max value

        :return: worst metric of all currently finalized trials
        :rtype: float
        """
        metric_history = self.get_metrics()
        return np.min(metric_history)

    def yworst(self):
        """Returns worst metric of all currently finalized trials

        Maximization problems are converted to minimization problems
        I.e. if the optimization direction is `max`, returns the negated min value

        :return: best metric of all currently finalized trials
        :rtype: float
        """
        metric_history = self.get_metrics()
        return np.max(metric_history)

    def ymean(self):
        """Returns best metric of all currently finalized trials

        Maximization problems are converted to minimization problems
        I.e. if the optimization direction is `max`, returns the mean of negated metrics

        :return: mean of all currently finalized trials metrics
        :rtype: float
        """
        metric_history = self.get_metrics()
        return np.mean(metric_history)

    def _log(self, msg):
        self.fd.write((msg + "\n").encode())
