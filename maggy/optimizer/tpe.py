import numpy as np
import statsmodels.api as sm
import scipy.stats as sps

from maggy.optimizer.abstractoptimizer import AbstractOptimizer
from maggy.trial import Trial

from hops import hdfs


class TPE(AbstractOptimizer):
    """
    I have the following class variables from the Abstract Optimizer

    self.searchspace = None # SearchSpace object containing the hp
    self.num_trials = None # just the number of trials int ?
    self.final_store = None # array of finished trials ( can we also access trials that are not finished )

    """

    def __init__(self):
        super().__init__()

        # initialize logger
        self.log_file = "hdfs:///Projects/playground/Logs/tpe.log"
        if not hdfs.exists(self.log_file):
            hdfs.dump("", self.log_file)
        self.fd = hdfs.open_file(self.log_file, flags="w")
        self._log("Initialized Logger")

        # keep track of the model (i.e the kernel density estimators l & g)
        self.model = None
        self.random_warmup_trials = []

        # meta hyper parameters
        # todo initialize them in constructor
        self.num_warmup_trails = 15
        self.gamma = 0.1
        self.num_samples = 24
        self.bw_estimation = "normal_reference"  # other options 'silvermann', 'scott'
        self.min_bw = 1e-3  # from HpBandSter
        self.bw_factor = 3  # higher values favor exploration
        self.random_fraction = 1 / 5  # todo findout good default

    # Abstract class methods

    # couldn't this be done in __init__
    def initialize(self):

        # initialize random trials
        random_samples = self.searchspace.get_random_parameter_values(
            self.num_warmup_trails
        )
        for parameters_dict in random_samples:
            self.random_warmup_trials.append(
                Trial(parameters_dict, trial_type="optimization")
            )

    def get_suggestion(self, trial=None):
        """Returns Trial instantiated with hparams that maximize the Expected Improvement"""

        try:
            self._log("Get Suggestion")

            # first sample randomly to warmup
            # maybe it would be better to check the if clause with existance of model or trial
            if self.random_warmup_trials:
                return self.random_warmup_trials.pop()

            self._log("Start updateing model")

            self._update_model()

            self._log("Model {}".format(str(self.model)))

            if not self.model or np.random.rand() < self.random_fraction:
                return Trial(self.searchspace.get_random_parameter_values(1)[0])

            best = -np.inf
            best_sample = None

            kde_good = self.model["good"]
            kde_bad = self.model["bad"]

            # loop through potential samples
            for sample in range(self.num_samples):
                idx = np.random.randint(0, len(kde_good.data))
                obs = kde_good.data[idx]
                # todo make definition of bounds in search space more explicit, below is interim solution
                bounds = np.array(
                    [spec[1] for hp, spec in self.searchspace.to_dict().items()]
                ).T  # ndarray with shape (2, n_hparams)
                sample_vector = []

                # loop through hparams
                for mean, bw, low, high in zip(obs, kde_good.bw, bounds[0], bounds[1]):
                    # clip by min bw and multiply by factor to favor more exploration
                    bw = max(bw, self.min_bw) * self.bw_factor

                    a, b = (
                        (low - mean) / bw,
                        (high - mean) / bw,
                    )  # see scipy documentation for more info
                    rv = sps.truncnorm(a, b, loc=mean, scale=bw)
                    sample_vector.append(rv)

                # calculate EI for current sample
                ei_val = TPE._calculate_ei(sample_vector, kde_good, kde_bad)

                if ei_val > best:
                    best = ei_val
                    best_sample = sample_vector

            # transform sample representation to dict
            # todo have general helper functions to change between representations
            hparam_names = list(self.searchspace.names().keys())
            best_sample_dict = {
                hparam_name: hparam
                for hparam_name, hparam in zip(hparam_names, best_sample)
            }

            self._log("Best Sample {}".format(best_sample_dict))

            return Trial(best_sample_dict)
        except BaseException as exc:
            self._log(exc)
            self.fd.flush()
            self.fd.close()

    def finalize_experiment(self, trials):
        return

    # optimizer specific methods

    def _update_model(self):
        """updates model based on all previous observations

        i.e. creating and storing kde for *good* and *bad* observations
        """

        # split trials in good and bad
        good_trials, bad_trials = self._split_trials()

        self._log("Split Good and Bad")

        # get list of hparams, each item of the list is one observation (list of all hparams)
        good_hparams = [list(trial.params.values()) for trial in good_trials]
        bad_hparams = [list(trial.params.values()) for trial in bad_trials]

        self.log("good: {}, bad: {}".format(good_hparams, bad_hparams))

        # todo consider case where we do not have enough observations ( return None )
        # → also see BOHB paper

        var_type = self._get_statsmodel_vartype()

        good_kde = sm.nonparametric.KDEMultivariate(
            data=good_hparams, var_type=var_type, bw=self.bw_estimation
        )
        bad_kde = sm.nonparametric.KDEMultivariate(
            data=bad_hparams, var_type=var_type, bw=self.bw_estimation
        )

        self.models = {"good": good_kde, "bad": bad_kde}

    def _split_trials(self):
        """splits trials in good and bad according to tpe algo

        :return: tuple with list of good trials and bad trials
        :rtype: (list[maggy.Trial], list[maggy.Trial])
        """

        # I need optimization direction here, for now assume minimize
        metric_history = np.array([trial.final_metric for trial in self.final_store])
        loss_idx_ascending = np.argsort(metric_history)
        n_good = int(np.ceil(self.gamma * len(metric_history)))

        self.log("Metric History: {}".format(metric_history))

        self._log(
            "loss_idx_ascending: {}, shape: {}".format(
                loss_idx_ascending, loss_idx_ascending.shape
            )
        )
        self._log("n_good: {}".format(n_good))

        self._log(
            "final store: {}, type: {}".format(self.final_store, type(self.final_store))
        )

        good_trails = self.final_store[np.sort(loss_idx_ascending[:n_good])]
        bad_trials = self.final_store[np.sort(loss_idx_ascending[n_good:])]

        return good_trails, bad_trials

    def _get_statsmodel_vartype(self):
        """Returns *statsmodel* type specifier string consisting of the types for each hparam of the searchspace , so for example 'ccuo'.

        :rtype: str
        """

        var_type_string = ""
        for hparam, vartype in self.sp.names.items():
            var_type_string += TPE._get_vartype(vartype)

        return var_type_string

    @staticmethod
    def _get_vartype(maggy_vartype):
        """Transforms Maggy vartype to statsmodel vartype, e.g. 'DOUBLE' → 'c'

        :param maggy_vartype: maggy type of hparam, e.g. 'DOUBLE'
        :type maggy_vartype: str
        :returns: corresponding vartype of statsmodel
        :rtype: str
        """
        if maggy_vartype == "DOUBLE":
            return "c"
        else:
            raise NotImplementedError("Only cont vartypes are implemented yer")

    @staticmethod
    def _calculate_ei(x, kde_good, kde_bad):
        """Returns Expected Improvement for given hparams

        :param x: list of hyperparameters
        :type x: list
        :param kde_good: kde of good observations
        :type kde_good: sm.KDEMultivariate
        :param kde_bad: pdf of kde of bad observations
        :type kde_bad: sm.KDEMultivariate of KDE instance
        :return: expected improvement
        :rtype: float
        """
        return max(1e-32, kde_good.pdf(x)) / max(kde_bad.pdf(x), 1e-32)

    def _log(self, msg):
        # self.fd = hdfs.open_file(self.log_file, flags="w")
        self.fd.write((msg + "\n").encode())
        # self.fd.flush()
        # self.fd.close()
