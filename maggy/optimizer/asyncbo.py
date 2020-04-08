import numpy as np

from maggy.optimizer.abstractoptimizer import AbstractOptimizer
from maggy.trial import Trial


class AsyncBayesianOptimization(AbstractOptimizer):
    """Base class for asynchronous bayesian optimization
    """

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

        self.warmup_trials_buffer = []  # keeps track of warmup trials

        self.busy_locations = []
        self.model = None

    def initialize(self):
        """initialize optimizer"""

        self.warmup_routine()
        self._init_model()  # Do I already need this here

    def get_suggestion(self, trial=None):
        """Returns next Trial or None when experiment is finished"""

        # todo put in try/catch block for the logger

        # check if experiment has finished
        if self._experiment_finished():
            return None

        # check if there are still Trials in the warmup buffer
        if self.warmup_trials_buffer:
            return self.warmup_trials_buffer.pop()

        # update model
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

    def sampling_routine(self):
        """Samples new config from model

        :return: hyperparameter config
        :rtype: dict

        """

    def warmup_routine(self):
        """implements logic for warming up bayesian optimization through random sampling"""

    def _init_model(self):
        """initializes the surrogate model of the gaussian process"""

    def _update_model(self):
        """updates the surrogate model with a new observation

        Question: can it update iteratively or do we need to generate a new model every time
        """

    def _acquisition_function(self):
        """calculates the utility for given point and surrogate"""

    def _maximize_acq_function(self):
        """maximizes acquisition function"""

    # helper functions

    def _experiment_finished(self):
        """checks if experiment is finished

        :return: True if experiment has finished, False else
        :rtype: bool

        In normal BO, experiment has finished when specified amount of trials have run,
        in BOHB/ASHA when all iterations have been finished
        """
