import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from skopt.learning.gaussian_process import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel
from skopt.learning.gaussian_process.kernels import Matern
from skopt.acquisition import _gaussian_acquisition
from skopt.acquisition import gaussian_acquisition_1D
from sklearn.base import clone

from maggy.optimizer.bayes.base import BaseAsyncBO

# todo what about noise in GP
# todo how and how often do the GP meta hparams get updated
# todo add documentation of how simple async bo works and what it is


class SimpleAsyncBO(BaseAsyncBO):
    """Base class for asynchronous bayesian optimization"""

    def __init__(self, impute_strategy="cl_min", **kwargs):
        """

        See docstring of `BaseAsyncBO` for more info on parameters of base class

        :param impute_strategy: Method to use as imputeing strategy in async bo.
                                Supported options are `"cl_min"`, `"cl_max"`, `"cl_mean"`.

                                - If set to `cl_x`, then constant liar strategy is used
                                  with lie objective value being minimum of observed objective
                                  values. `"cl_mean"` and `"cl_max"` means mean and max of values
                                  respectively.
                                - If set to `kb`, then kriging believer strategy is used with lie objective value being
                                  the models predictive mean

                                For more information on strategies see:

                                https://www.cs.ubc.ca/labs/beta/EARG/stack/2010_CI_Ginsbourger-ParallelKriging.pdf
        :type impute_strategy: str
        """
        super().__init__(**kwargs)

        # configure impute strategy

        allowed_impute_strategies = ["cl_min", "cl_max", "cl_mean", "kb"]
        if impute_strategy not in allowed_impute_strategies:
            raise ValueError(
                "expected impute_strategy to be in {}, got {}".format(
                    allowed_impute_strategies, impute_strategy
                )
            )
        self.impute_strategy = impute_strategy

    def sampling_routine(self, budget=0):
        self._log("Start sampling routine from model with budget {}".format(budget))

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

        values = _gaussian_acquisition(
            X=X,
            model=self.models[budget],
            y_opt=self.ybest(budget),
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
            # todo does it make sense to use ybest of budget only
            y_best = self.ybest(budget)
            for x in x0:
                # todo evtl. use Parallel / delayed like skopt
                # bounds of transformed hparams are always [0.0,1.0] ( if categorical encodings get normalized,
                # which is the case here )
                result = fmin_l_bfgs_b(
                    func=gaussian_acquisition_1D,
                    x0=x,
                    args=(
                        self.models[budget],
                        y_best,
                        self.acq_fun,
                        self.acq_func_kwargs,
                    ),
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

        # get hparams and final metrics of finished trials combined with busy locations
        Xi = self.get_hparams_array(include_busy_locations=True, budget=budget)
        yi = self.get_metrics_array(include_busy_locations=True, budget=budget)

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
