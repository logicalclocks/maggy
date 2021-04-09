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

from abc import ABC
from abc import abstractmethod

import numpy as np
from skopt.acquisition import _gaussian_acquisition
from skopt.acquisition import gaussian_acquisition_1D


class AbstractAcquisitionFunction(ABC):
    @staticmethod
    @abstractmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs=None):
        """evaluates acquisition function at given points

        :param X: Values where the acquisition function should be computed. shape = (n_locations, n_hparams)
        :type X: np.ndarray
        :param surrogate_model: the surrogate model of the bayesian optimizer.
        :type surrogate_model: GaussianProcessRegressor
        :param y_opt: currently best observed value
        :type y_opt: float
        :param acq_func_kwargs: additional arguments for the acquisition function
        :type acq_func_kwargs: dict|None
        :return: Acquisition function values computed at X. shape = (n_locations,)
        :rtype: np.ndarray
        """
        pass

    @staticmethod
    @abstractmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs=None):
        """A wrapper around the acquisition function that is called by fmin_l_bfgs_b.
           This is because lbfgs allows only 1-D input.

        :param x: value where acquisition function should be evaluated. shape=(n_hparams, )
        :type x: np.ndarray
        :param surrogate_model: the surrogate model of the bayesian optimizer.
        :type surrogate_model: GaussianProcessRegressor
        :param y_opt: currently best observed value
        :type y_opt: float
        :param acq_func_kwargs: additional arguments for the acquisition function
        :type acq_func_kwargs: dict|None
        :return: tuple containing two arrays. the first holds the evaluated values of the acquisition function at value
                 x; shape = (1,) . the second holds the gradients; shape = (n_hparams,).
        :rtype: tuple
        """
        pass

    def name(self):
        return str(self.__class__.__name__)


class GaussianProcess_EI(AbstractAcquisitionFunction):
    """xi in acq_func_kwargs"""

    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs=None):
        return _gaussian_acquisition(
            X=X,
            model=surrogate_model,
            y_opt=y_opt,
            acq_func="EI",
            acq_func_kwargs=acq_func_kwargs,
        )

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs=None):
        return gaussian_acquisition_1D(
            X=x,
            model=surrogate_model,
            y_opt=y_opt,
            acq_func="EI",
            acq_func_kwargs=acq_func_kwargs,
        )


class GaussianProcess_PI(AbstractAcquisitionFunction):
    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs=None):
        return _gaussian_acquisition(
            X=X,
            model=surrogate_model,
            y_opt=y_opt,
            acq_func="PI",
            acq_func_kwargs=acq_func_kwargs,
        )

    @staticmethod
    def evaluate_1_d(X, surrogate_model, y_opt, acq_func_kwargs=None):
        return gaussian_acquisition_1D(
            X=X,
            model=surrogate_model,
            y_opt=y_opt,
            acq_func="PI",
            acq_func_kwargs=acq_func_kwargs,
        )


class GaussianProcess_LCB(AbstractAcquisitionFunction):
    """kappa in acq_func_kwargs"""

    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs=None):
        return _gaussian_acquisition(
            X=X,
            model=surrogate_model,
            y_opt=y_opt,
            acq_func="LCB",
            acq_func_kwargs=acq_func_kwargs,
        )

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs=None):
        return gaussian_acquisition_1D(
            X=x,
            model=surrogate_model,
            y_opt=y_opt,
            acq_func="LCB",
            acq_func_kwargs=acq_func_kwargs,
        )


class GaussianProcess_UCB(AbstractAcquisitionFunction):
    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs=None):
        raise NotImplementedError

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs=None):
        raise NotImplementedError


class TPE_EI(AbstractAcquisitionFunction):
    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs=None):
        raise NotImplementedError

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs=None):
        raise NotImplementedError


class AsyTS(AbstractAcquisitionFunction):
    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs=None):
        return surrogate_model.sample_y(X).reshape(
            X.shape[0],
        )

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs=None):
        """A wrapper around the acquisition function that is called by fmin_l_bfgs_b.
           This is because lbfgs allows only 1-D input.

        :param x: value where acquisition function should be evaluated. shape=(n_hparams, )
        :type x: np.ndarray
        :param surogate_model: the surrogate model of the bayesian optimizer.
        :type surogate_model: GaussianProcessRegressor
        :param y_opt: currently best observed value
        :type y_opt: float
        :param acq_func_kwargs: additional arguments for the acquisition function
        :type acq_func_kwargs: dict|None
        :return: values of the acquisition function at value x. shape = (1,)
        :rtype: np.ndarray
        """
        return surrogate_model.sample_y(np.expand_dims(x, axis=0)).reshape(
            1,
        )


class HLP(AbstractAcquisitionFunction):
    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs=None):
        raise NotImplementedError

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs=None):
        raise NotImplementedError
