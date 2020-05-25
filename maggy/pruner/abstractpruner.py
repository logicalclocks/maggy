from abc import ABC, abstractmethod

LOCAL = False  # if set to true, can be run locally without maggy integration
# todo all the `if not LOCAL:` clause can be removed after tested locally

if not LOCAL:
    from hops import hdfs


class AbstractPruner(ABC):
    def __init__(self, trial_metric_getter):

        """
        :param trial_metric_getter: a function that returns a dict with `trial_id` as key and `metric` as value
                             with the lowest metric being the "best"
                             It's only argument is `trial_ids`, it can be either str of single trial or list of trial ids
        :type trial_metric_getter: function
        """

        if not LOCAL:
            # configure logger
            self.log_file = "hdfs:///Projects/{}/Experiment_Logs/pruner_{}_{}.log".format(
                hdfs.project_name(),
                self.name(),
                trial_metric_getter.__self__.__class__.__name__ if not LOCAL else "",
            )
            if not hdfs.exists(self.log_file):
                hdfs.dump("", self.log_file)
            self.fd = hdfs.open_file(self.log_file, flags="w")
            self._log("Initialized Logger")

        self.trial_metric_getter = trial_metric_getter

    @abstractmethod
    def pruning_routine(self):
        """
        runs pruning routine.
        interface top `optimizer`
        """
        pass

    @abstractmethod
    def report_trial(self):
        """
        hook for reporting trial id of created trial from optimizer to pruner
        """
        pass

    @abstractmethod
    def finished(self):
        """
        checks if experiment is finished
        """
        pass

    @abstractmethod
    def num_trials(self):
        """
        calculates the number of trials in the experiment

        :return: number of trials
        :rtype: int
        """

    def name(self):
        return str(self.__class__.__name__)

    def _log(self, msg):
        if not LOCAL:
            if not self.fd.closed:
                self.fd.write((msg + "\n").encode())
        print(msg, "\n")

    def _close_log(self):
        if not LOCAL:
            if not self.fd.closed:
                self.fd.flush()
                self.fd.close()
