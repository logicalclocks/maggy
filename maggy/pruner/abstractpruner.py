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

from abc import ABC, abstractmethod
from datetime import datetime

from maggy.core.environment.singleton import EnvSing


class AbstractPruner(ABC):
    def __init__(self, trial_metric_getter):
        """
        :param trial_metric_getter: a function that returns a dict with `trial_id` as key and `metric` as value
                             with the lowest metric being the "best"
                             It's only argument is `trial_ids`, it can be either str of single trial or list of trial ids
        :type trial_metric_getter: function
        """

        self.trial_metric_getter = trial_metric_getter

        # logger variables
        self.log_file = None
        self.fd = None

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

    def initialize_logger(self, exp_dir):
        """Initialize logger of optimizer

        :param exp_dir: path of experiment directory
        :rtype exp_dir: str
        """
        env = EnvSing.get_instance()
        # configure logger
        self.log_file = exp_dir + "/pruner.log"

        if not env.exists(self.log_file):
            env.dump("", self.log_file)
        self.fd = env.open_file(self.log_file, flags="w")
        self._log("Initialized Pruner Logger")

    def _log(self, msg):
        if self.fd and not self.fd.closed:
            msg = datetime.now().isoformat() + ": " + str(msg)
            self.fd.write(EnvSing.get_instance().str_or_byte(msg + "\n"))

    def _close_log(self):
        if not self.fd.closed:
            self.fd.flush()
            self.fd.close()
