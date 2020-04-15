import traceback

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from skopt.learning.gaussian_process import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel
from skopt.learning.gaussian_process.kernels import Matern
from skopt.acquisition import _gaussian_acquisition
from skopt.acquisition import gaussian_acquisition_1D
from sklearn.base import clone

from maggy.optimizer.abstractoptimizer import AbstractOptimizer
from maggy.trial import Trial

from hops import hdfs


# todo which methods should be private


class AsyncBayesianOptimization(AbstractOptimizer):
    """Base class for asynchronous bayesian optimization

    todo add default init values
    """

    def __init__(
        self,
        num_warmup_trials,
        random_fraction,
        acq_fun,
        acq_fun_kwargs,
        acq_optimizer,
        acq_optimizer_kwargs,
        pruner,
        pruner_kwargs,
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
        self.cl_strategy = "cl_min"

        # configure logger

        self.log_file = (
            "hdfs:///Projects/demo_deep_learning_admin000/Logs/asyncbo.log"  # todo
        )
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
                return self.warmup_trial_buffer.pop()

            # update model with latest observations
            self._update_model()

            # in case there is no model yet or random fraction applies, sample randomly
            # todo in case of BOHB/ASHA model is a dict, maybe it should be dict for every case
            if not self.model or np.random.rand() < self.random_fraction:
                hparams = self.searchspace.get_random_parameter_values(1)[0]
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

    def sampling_routine(self, impute_busy=True):
        """Samples new config from model

        # todo add description of what is happening


        :return: hyperparameter config
        :rtype: dict

        """
        # even with BFGS as optimizer we want to sample a large number
        # of points and then pick the best ones as starting points
        random_hparams = self.searchspace.get_random_parameter_values(self.n_points)
        random_hparams_list = np.array(
            [self.searchspace.dict_to_list(hparams) for hparams in random_hparams]
        )

        # transform configs
        X = np.apply_along_axis(
            self.searchspace.transform,
            1,
            random_hparams_list,
            normalize_categorical=True,
        )

        # todo convert max problem e.g. accuracy to min problem → probably not the right place to implement it here
        #  though
        values = _gaussian_acquisition(
            X=X,
            model=self.model,
            y_opt=self.ymin(),
            acq_func=self.acq_fun,
            acq_func_kwargs=self.acq_func_kwargs,
        )

        # Find the minimum of the acquisition function by randomly
        # sampling points from the space
        if self.acq_optimizer == "sampling":
            next_x = X[np.argmin(values)]

        # Use BFGS to find the mimimum of the acquisition function, the
        # minimization starts from `n_restarts_optimizer` different
        # points and the best minimum is
        elif self.acq_optimizer == "lbfgs":
            x0 = X[np.argsort(values)[: self.n_restarts_optimizer]]

            results = []
            for x in x0:
                # todo evtl. use Parallel / delayed like skopt
                # bounds of transformed hparams are always [0.0,1.0] ( if categorical encodings get normalized,
                # which is the case here )
                result = fmin_l_bfgs_b(
                    func=gaussian_acquisition_1D,
                    x0=x,
                    args=(self.model, self.ymin(), self.acq_fun, self.acq_func_kwargs),
                    bounds=[(0.0, 1.0) for _ in self.searchspace.values()],
                    approx_grad=False,
                    maxiter=20,
                )
                results.append(result)

            cand_xs = np.array([r[0] for r in results])
            cand_acqs = np.array([r[1] for r in results])
            next_x = cand_xs[np.argmin(cand_acqs)]

        # lbfgs should handle this but just in case there are
        # precision errors.
        next_x = np.clip(next_x, 0.0, 1.0)

        # transform back to original representation
        next_x = self.searchspace.inverse_transform(
            next_x, normalize_categorical=True
        )  # is array [-3,3,"blue"]

        self._log("Next config to evaluate: {}".format(next_x))

        # add next_x to busy locations and impute metric with constant liar
        if impute_busy:
            if self.cl_strategy == "cl_min":
                cl = self.ymin()
            elif self.cl_strategy == "cl_max":
                cl = self.ymax()
            elif self.cl_strategy == "cl_mean":
                cl = self.ymean()
            else:
                raise NotImplementedError(
                    "cl_strategy {} is not implemented, please choose from ('cl_min', 'cl_max', "
                    "'cl_mean')"
                )
            self.busy_locations.append({"params": next_x, "metric": cl})

        self._log("busy_locations: {}".format(self.busy_locations))

        # convert list to dict representation
        hparam_dict = self.searchspace.list_to_dict(next_x)

        return hparam_dict

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

        # add configs to trial buffer
        for hparams in warmup_configs:
            self.warmup_trial_buffer.append(Trial(hparams, trial_type="optimization"))

    def init_model(self):
        """initializes the surrogate model of the gaussian process

        the model gets created with the right parameters, but is not fit with any data yet. the `base_model` will be
        cloned in `update_model` and fit with observation data
        """

        n_dims = len(self.searchspace.keys())

        # ToDo, find out why I need this → skopt/utils.py line 377
        cov_amplitude = ConstantKernel(1.0, (0.01, 1000.0))

        # ToDo implement special case if all dimesnsions are catigorical --> skopt/utils.py l. 378ff
        # ToDo compare the initialization of kernel parameters with other frameworks
        other_kernel = Matern(
            length_scale=np.ones(n_dims),
            length_scale_bounds=[(0.01, 100)] * n_dims,
            nu=2.5,
        )
        base_model = GaussianProcessRegressor(
            kernel=cov_amplitude * other_kernel,
            normalize_y=True,
            noise="gaussian",
            n_restarts_optimizer=2,
        )
        self.base_model = base_model

    def _update_model(self):
        """update surrogate model with new observations

        Use observations of finished trials + liars from busy trials to build model.
        Only build model when there are at least as many observations as hyperparameters
        """
        # check if enough observations available for model building
        if len(self.searchspace.keys()) > len(self.final_store):
            self._log("Not enough observations available to build yet")
            return

        # create model without any data
        model = clone(self.base_model)

        # get hparams and final metrics of finished trials
        Xi = np.array(
            [self.searchspace.dict_to_list(trial.params) for trial in self.final_store]
        )
        yi = np.array([trial.final_metric for trial in self.final_store])

        self._log("Xi: {}".format(Xi))

        # get locations of busy trials and imputed liars
        if len(self.busy_locations):

            Xi_busy = np.array([location["params"] for location in self.busy_locations])
            yi_busy = np.array([location["metric"] for location in self.busy_locations])

            # join observations with busy locations
            Xi = np.vstack((Xi, Xi_busy))
            yi = np.vstack((yi, yi_busy))

            self._log("Xi_busy: {}".format(Xi_busy))
            self._log("Xi_combined: {}".format(Xi))

        # transform hparam values
        Xi_transform = np.apply_along_axis(
            self.searchspace.transform, 1, Xi, normalize_categorical=True
        )

        self._log("Xi_transform: {}".format(Xi_transform))

        # fit model with data
        model.fit(Xi_transform, yi)

        # set current model to the fitted estimator
        self.model = model

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
                  to compute but at t,he same would ensure consistency → ask Moritz

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

    def ymin(self):
        """
        :return: min metric of all currently finalized trials
        :rtype: float
        """
        metric_history = np.array([trial.final_metric for trial in self.final_store])
        return np.min(metric_history)

    def ymax(self):
        """
        :return: max metric of all currently finalized trials
        :rtype: float
        """
        metric_history = np.array([trial.final_metric for trial in self.final_store])
        return np.max(metric_history)

    def ymean(self):
        """
        :return: mean of all currently finalized trials metrics
        :rtype: float
        """
        metric_history = np.array([trial.final_metric for trial in self.final_store])
        return np.mean(metric_history)

    def _log(self, msg):
        self.fd.write((msg + "\n").encode())
