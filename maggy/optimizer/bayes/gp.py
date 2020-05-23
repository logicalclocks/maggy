import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from skopt.learning.gaussian_process import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel
from skopt.learning.gaussian_process.kernels import Matern
from sklearn.base import clone

from maggy.optimizer.bayes.base import BaseAsyncBO
from maggy.optimizer.bayes.acquisitions import (
    GaussianProcess_EI,
    GaussianProcess_LCB,
    GaussianProcess_UCB,
    GaussianProcess_PI,
    AsyTS,
    HLP,
)

# todo what about noise in GP
# todo how and how often do the GP meta hparams get updated (lenght_scale etc.) so far length scale is set to 1
# todo add documentation of how simple async bo works and what it is
# todo explain gp


class GP(BaseAsyncBO):
    """Base class for asynchronous bayesian optimization"""

    def __init__(
        self,
        async_strategy="impute",
        impute_strategy="cl_min",
        acq_fun=None,
        acq_fun_kwargs=None,
        acq_optimizer="lbfgs",
        acq_optimizer_kwargs=None,
        **kwargs
    ):
        """

        See docstring of `BaseAsyncBO` for more info on parameters of base class

        :param async_strategy: strategy to encourage diversity when sampling. Can take following values
                               todo explain the strategies
                               - `"impute"`
                               - `"asy_ts"`
                               - `"playbook"`

        :type async_strategy: str
        :param impute_strategy: Method to use as imputeing strategy in async bo, if async_strategy is `"impute"`
                                Supported options are `"cl_min"`, `"cl_max"`, `"cl_mean"`, `"kb"`.

                                - If set to `cl_x`, then constant liar strategy is used
                                  with lie objective value being minimum of observed objective
                                  values. `"cl_mean"` and `"cl_max"` means mean and max of values
                                  respectively.
                                - If set to `kb`, then kriging believer strategy is used with lie objective value being
                                  the models predictive mean

                                For more information on strategies see:

                                https://www.cs.ubc.ca/labs/beta/EARG/stack/2010_CI_Ginsbourger-ParallelKriging.pdf
        :type impute_strategy: str|None
        :param acq_fun: Function to minimize over the posterior distribution. Can take different values depending on
                        chosen `async_strategy`. If None, the correct default is chosen
                        - impute
                            - `"EI"` for negative expected improvement.
                            - `"LCB"` for lower confidence bound.
                            - `"PI"` for negative probability of improvement.
                        - asy_ts
                            - `"AsyTS"` in async thompson sampling, the acquisition function is replaced by thompson sampling
                        - playbook
                            - `"UCB"` for upper confidence bound
                            - todo EI possible as well ?
        :type acq_fun: str|None
        :param acq_fun_kwargs: Additional arguments to be passed to the acquisition function.
        :type acq_fun_kwargs: dict|None
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

        Attributes
        ----------

        base_model (any): estimator that has not been fit on any data.
        """
        super().__init__(**kwargs)

        # validations

        # allowed combinations of async strategies and acquisition functions
        allowed_combinations = {
            "impute": {
                "EI": GaussianProcess_EI,
                "LCB": GaussianProcess_LCB,
                "UCB": GaussianProcess_UCB,
                "PI": GaussianProcess_PI,
            },
            "asy_ts": {"AsyTS": AsyTS},
            "playbook": {"UCB": HLP},
        }
        if async_strategy not in allowed_combinations.keys():
            raise ValueError(
                "Expected async_strategy to be in {} with GP as surrogate, got {}".format(
                    list(allowed_combinations.keys()), async_strategy
                )
            )

        if acq_fun not in allowed_combinations[async_strategy] and acq_fun is not None:
            raise ValueError(
                "Expected acq_fun to be in {} with GP as surrogate and {} as async_strategy, got {}".format(
                    list(allowed_combinations[async_strategy].keys()),
                    async_strategy,
                    acq_fun,
                )
            )

        # async_strategy
        self.async_strategy = async_strategy

        # configure acquisition function
        if acq_fun is None:
            # default acq_fun is the first in the dict
            acq_fun = list(allowed_combinations[async_strategy].keys())[0]
        self.acq_fun = allowed_combinations[self.async_strategy][acq_fun]()
        self.acq_func_kwargs = acq_fun_kwargs

        # configure acquisiton function optimizer
        allowed_acq_opt = ["sampling", "lbfgs"]
        if acq_optimizer not in allowed_acq_opt:
            raise ValueError(
                "expected acq_optimizer to be in {}, got {}".format(
                    allowed_acq_opt, acq_optimizer
                )
            )
        self.acq_optimizer = acq_optimizer
        if acq_optimizer_kwargs is None:
            acq_optimizer_kwargs = dict()

        # todo evtl verallgemeinern und oben via dict klären, vgl. acq functions
        if self.async_strategy == "asy_ts":
            # default value is 100 and max value is 1000 for asy ts
            self.n_points = np.clip(acq_optimizer_kwargs.get("n_points", 100), 10, 1000)
        else:
            self.n_points = acq_optimizer_kwargs.get("n_points", 10000)
        self.n_restarts_optimizer = acq_optimizer_kwargs.get("n_restarts_optimizer", 5)
        self.acq_optimizer_kwargs = acq_optimizer_kwargs

        # configure impute strategy
        if self.async_strategy == "impute":
            allowed_impute_strategies = ["cl_min", "cl_max", "cl_mean", "kb"]
            if impute_strategy not in allowed_impute_strategies:
                raise ValueError(
                    "expected impute_strategy to be in {}, got {}".format(
                        allowed_impute_strategies, impute_strategy
                    )
                )
            self.impute_strategy = impute_strategy
            # sample from the model that has same budget as the trial and not always from largest model available
            if self.pruner:
                self.max_model = False

        # estimator that has not been fit on any data.
        self.base_model = None

        # write logs
        self._log(
            "Acquisition Function: {}, Async Strategy: {}".format(
                self.acq_fun.name(), self.async_strategy
            )
        )
        if self.async_strategy == "impute":
            self._log("Impute Strategy: {}".format(self.impute_strategy))

    def sampling_routine(self, budget=0):
        self._log("Start sampling routine from model with budget {}".format(budget))

        # even with BFGS as optimizer we want to sample a large number
        # of points and then pick the best ones as starting points
        random_hparams = self.searchspace.get_random_parameter_values(self.n_points)
        random_hparams_list = np.array(
            [self.searchspace.dict_to_list(hparams) for hparams in random_hparams]
        )

        # todo does it make sense to use ybest of budget only
        y_opt = self.ybest(budget)

        # transform configs
        X = np.apply_along_axis(
            self.searchspace.transform,
            1,
            random_hparams_list,
            normalize_categorical=True,
        )

        values = self.acq_fun.evaluate(
            X=X,
            surrogate_model=self.models[budget],
            y_opt=y_opt,
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
            # in asynchronous ts the acquisition function is not returning the gradient, hence we need to approximate
            approx_grad = True if self.async_strategy == "asy_ts" else False

            x0 = X[np.argsort(values)[: self.n_restarts_optimizer]]

            results = []
            for x in x0:
                # todo evtl. use Parallel / delayed like skopt
                # bounds of transformed hparams are always [0.0,1.0] ( if categorical encodings get normalized,
                # which is the case here )
                result = fmin_l_bfgs_b(
                    func=self.acq_fun.evaluate_1_d,
                    x0=x,
                    args=(self.models[budget], y_opt, self.acq_func_kwargs,),
                    bounds=[(0.0, 1.0) for _ in self.searchspace.values()],
                    approx_grad=approx_grad,
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

        # convert list to dict representation
        hparam_dict = self.searchspace.list_to_dict(next_x)

        return hparam_dict

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

    def update_model(self, budget=0):
        """update surrogate model with new observations

        Use observations of finished trials + liars from busy trials to build model.
        Only build model when there are at least as many observations as hyperparameters
        """
        self._log("Start updateing model with budget {} \n".format(budget))
        for key, val in self.trial_store.items():
            self._log("{}: {} \n".format(key, val.params))

        # check if enough observations available for model building
        if len(self.searchspace.keys()) > len(self.get_metrics_array(budget=budget)):
            self._log(
                "Not enough observations available to build with budget {} yet. At least {} needed, got {}".format(
                    budget,
                    len(self.searchspace.keys()),
                    len(self.get_metrics_array(budget=budget)),
                )
            )
            return

        # create model without any data
        model = clone(self.base_model)

        # get hparams and final metrics of finished trials combined with busy locations if imputeing stratgey
        include_busy_locations = True if self.async_strategy == "impute" else False
        Xi = self.get_hparams_array(
            include_busy_locations=include_busy_locations, budget=budget
        )
        yi = self.get_metrics_array(
            include_busy_locations=include_busy_locations, budget=budget
        )

        # transform hparam values
        Xi_transform = np.apply_along_axis(
            self.searchspace.transform, 1, Xi, normalize_categorical=True
        )

        # self._log("Xi: {}".format(Xi))
        # self._log("Xi_transform: {}".format(Xi_transform))
        # self._log("yi: {}".format(yi))

        # fit model with data
        model.fit(Xi_transform, yi)

        self._log("Fitted Model with data")

        # update model of budget
        self.models[budget] = model

    def impute_metric(self, hparams, budget=0):
        """calculates the value of the imputed metric for hparams of a currently evaluating trial.

        This is the core of the async strategy of `constant_liar` and `kriging believer`, i.e. currently evaluating
        trial are given a imputed metric and the model is updated with the hparam config and imputed metric so it does
        not yield the same hparam conig again when maximizing the acquisition function, hence encourages diversity when
        choosing the next hparam config to evaluate.

        :param hparams: hparams dict of a currently evaluating trial (Trial.params)
        :type hparams: dict
        :param budget: budget of the model that sampled the hparam config
        :param budget: int
        :return: imputed metric
        :rtype: float
        """

        if self.impute_strategy == "cl_min":
            imputed_metric = self.ybest(budget)
        elif self.impute_strategy == "cl_max":
            imputed_metric = self.yworst(budget)
        elif self.impute_strategy == "cl_mean":
            imputed_metric = self.ymean(budget)
        elif self.impute_strategy == "kb":
            x = self.searchspace.transform(
                hparams=self.searchspace.dict_to_list(hparams),
                normalize_categorical=True,
            )
            imputed_metric = self.models[budget].predict(np.array(x).reshape(1, -1))[0]
        else:
            raise NotImplementedError(
                "cl_strategy {} is not implemented, please choose from ('cl_min', 'cl_max', "
                "'cl_mean')"
            )

        # if the optimization direction is max the above strategies yield the negated value of the metric,
        # return the original metric value
        if self.direction == "max":
            imputed_metric = -imputed_metric

        return imputed_metric
