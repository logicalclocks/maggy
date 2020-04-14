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


# todo which methods should be private


class AsyncBayesianOptimization(AbstractOptimizer):
    """Base class for asynchronous bayesian optimization"""

    def __init__(
        self,
        num_warmup_trials,
        random_fraction,
        acq_func,
        acq_func_kwargs,
        acq_optimizer,
        acq_optimizer_kwargs,
        pruner,
        pruner_kwargs,
    ):
        super().__init__()

        # from AbstractOptimizer
        # self.final_store # dict of trials
        # selt.trial_store # list of trials → all trials or only unfinished trials ??
        # self.direction
        # self.num_trials
        # self.searchspace

        self.num_warmup_trials = num_warmup_trials
        self.random_fraction = random_fraction
        self.initial_sampling = "random"  # other options could be latin hypercube
        self.acq_fun = acq_func  # calculates the utility for given point and surrogate
        self.acq_func_kwargs = acq_func_kwargs
        self.acq_optimizer = acq_optimizer  # sampling/lbfgs
        self.acq_optimizer_kwargs = acq_optimizer_kwargs
        self.pruner = (
            pruner  # class vs. instance vs. string ?? → same discussion for acq_fun
        )
        self.pruner_kwargs = pruner_kwargs

        self.warmup_trial_buffer = []  # keeps track of warmup trials

        self.busy_locations = (
            []
        )  # each busy location is a dict {"params": hparams_list, "metric": liar}
        self.base_estimator = None  # estimator that has not been fit on any data.
        self.model = None  # fitted model of the estimator
        self.y_opt = None  # currently best metric # todo, evtl easier to always compute on the go

    def initialize(self):
        """initialize optimizer"""

        self.warmup_routine()
        self._init_model()  # todo:  do I already need this here ??

    def get_suggestion(self, trial=None):
        """Returns next Trial or None when experiment is finished"""

        # todo put in try/catch block for the logger. In general add logger statements where needed

        # check if experiment has finished
        if self._experiment_finished():
            return None

        # check if there are still Trials in the warmup buffer
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

    def finalize_experiment(self, trials):
        return

    def sampling_routine(self, impute_busy=False):
        """Samples new config from model

        # todo add description of what is happening

        :return: hyperparameter config
        :rtype: dict

        """

        # todo implement helpers for: transforming from dict to array and for hparam transforms
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

        # todo convert max problem e.g. accuracy to min problem → probably not the right place to implement it here though
        # y_opt is best value (i.e. min )
        values = _gaussian_acquisition(
            X=X,
            model=self.model,
            y_opt=self.y_opt,
            acq_func=self.acq_func,
            acq_func_kwargs=self.acq_func_kwargs,
        )

        # Find the minimum of the acquisition function by randomly
        # sampling points from the space
        if self.acq_optimizer == "sampling":
            next_x = X[np.argmin(values)]

        # Use BFGS to find the mimimum of the acquisition function, the
        # minimization starts from `n_restarts_optimizer` different
        # points and the best minimum is
        # todo insert link to lbfgs
        elif self.acq_optimizer == "lbfgs":
            x0 = X[
                np.argsort(values)[: self.acq_optimizer_kwargs["n_restarts_optimizer"]]
            ]

            results = []
            for x in x0:
                # todo evtl. use Parallel / delayed like skopt
                # bounds of transformed hparams are always [0.0,1.0] ( if categorical encodings get normalized,
                # which is the case here )
                result = fmin_l_bfgs_b(
                    func=gaussian_acquisition_1D,
                    x0=x,
                    args=(self.model, self._ymin(), self.acq_fun, self.acq_func_kwargs),
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

        return self.searchspace.list_to_dict(next_x)

    def warmup_routine(self):
        """implements logic for warming up bayesian optimization through random sampling by adding hparam configs to
        `warmup_trials_buffer` """

        # generate random hparam configs
        warmup_configs = self.searchspace.get_random_parameter_values(
            self.num_warmup_trials
        )

        # add configs to trial buffer
        for counter, parameters_dict in enumerate(warmup_configs):
            self.warmup_trial_buffer.append(
                Trial(parameters_dict, trial_type="optimization")
            )

    def _init_model(self):
        """initializes the surrogate model of the gaussian process"""

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

        # ToDo naming
        base_estimator = GaussianProcessRegressor(
            kernel=cov_amplitude * other_kernel,
            normalize_y=True,
            noise="gaussian",
            n_restarts_optimizer=2,
        )

        self.model = base_estimator

    def _update_model(self):
        """update surrogate model with new observations

        Use observations of finished trials + liars from busy trials to build model

        todo: can it update iteratively or do we need to generate a new model every time ?
        """
        # create model without any data
        est = clone(self.base_estimator)

        # get hparams and final metrics of finished trials
        Xi = np.array(
            [self.searchspace.dict_to_list(trial.params) for trial in self.final_store]
        )
        yi = np.array([trial.final_metric for trial in self.final_store])

        # get locations of busy trials and imputed liars
        Xi_busy = np.array([location["params"] for location in self.busy_locations])
        yi_busy = np.array([location["metric"] for location in self.busy_locations])

        # join observations with busy locations
        Xi_combined = np.vstack((Xi, Xi_busy))
        yi_combined = np.vstack((yi, yi_busy))

        # transform hparam values
        Xi_transform = np.apply_along_axis(
            self.searchspace.transform, 1, Xi_combined, normalize_categorical=True
        )

        # todo log if transform worked properly

        # fit model with data
        est.fit(Xi_transform, yi_combined)

        # set current model to the fitted estimator
        self.model = est

    def _acquisition_function(self):
        """calculates the utility for given point and surrogate"""

    def _maximize_acq_function(self):
        """maximizes acquisition function"""

    def _experiment_finished(self):
        """checks if experiment is finished

        :return: True if experiment has finished, False else
        :rtype: bool

        In normal BO, experiment has finished when specified amount of trials have run,
        in BOHB/ASHA when all iterations have been finished
        """

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
