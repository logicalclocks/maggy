from abc import ABC
from abc import abstractmethod

from skopt.acquisition import _gaussian_acquisition
from skopt.acquisition import gaussian_acquisition_1D


class AbstractAcquisitionFunction(ABC):
    @staticmethod
    @abstractmethod
    def evaluate(X, surogate_model, y_opt, acq_func_kwargs):
        pass

    @staticmethod
    @abstractmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs):
        pass


class GaussianProcess_EI(AbstractAcquisitionFunction):
    """xi in acq_func_kwargs"""

    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs):
        return _gaussian_acquisition(
            X=X,
            model=surrogate_model,
            y_opt=y_opt,
            acq_func="EI",
            acq_func_kwargs=acq_func_kwargs,
        )

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs):
        return gaussian_acquisition_1D(
            X=x,
            model=surrogate_model,
            y_opt=y_opt,
            acq_func="EI",
            acq_func_kwargs=acq_func_kwargs,
        )


class GaussianProcess_PI(AbstractAcquisitionFunction):
    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs):
        return _gaussian_acquisition(
            X=X,
            model=surrogate_model,
            y_opt=y_opt,
            acq_func="PI",
            acq_func_kwargs=acq_func_kwargs,
        )

    @staticmethod
    def evaluate_1_d(X, surrogate_model, y_opt, acq_func_kwargs):
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
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs):
        return _gaussian_acquisition(
            X=X,
            model=surrogate_model,
            y_opt=y_opt,
            acq_func="LCB",
            acq_func_kwargs=acq_func_kwargs,
        )

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs):
        return gaussian_acquisition_1D(
            X=x,
            model=surrogate_model,
            y_opt=y_opt,
            acq_func="LCB",
            acq_func_kwargs=acq_func_kwargs,
        )


class GaussianProcess_UCB(AbstractAcquisitionFunction):
    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs):
        raise NotImplementedError

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs):
        raise NotImplementedError


class TPE_EI(AbstractAcquisitionFunction):
    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs):
        raise NotImplementedError

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs):
        raise NotImplementedError


class AsyTS(AbstractAcquisitionFunction):
    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs):
        raise NotImplementedError

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs):
        raise NotImplementedError


class HLP(AbstractAcquisitionFunction):
    @staticmethod
    def evaluate(X, surrogate_model, y_opt, acq_func_kwargs):
        raise NotImplementedError

    @staticmethod
    def evaluate_1_d(x, surrogate_model, y_opt, acq_func_kwargs):
        raise NotImplementedError
