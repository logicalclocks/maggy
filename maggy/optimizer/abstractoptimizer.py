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

from hops import hdfs


class AbstractOptimizer(ABC):
    def __init__(self):
        self.searchspace = None
        self.num_trials = None
        self.trial_store = None
        self.final_store = None
        self.direction = None
        self.pruner = None

        # logger variables
        self.log_file = None
        self.fd = None

    @abstractmethod
    def initialize(self):
        """
        A hook for the developer to initialize the optimizer.
        """
        pass

    @abstractmethod
    def get_suggestion(self, trial=None):
        """
        Return a `Trial` to be assigned to an executor, or `None` if there are
        no trials remaining in the experiment.

        :param trial: last finished trial by an executor

        :rtype: Trial or None
        """
        pass

    @abstractmethod
    def finalize_experiment(self, trials):
        """
        This method will be called before finishing the experiment. Developers
        can implement this method e.g. for cleanup or extra logging.
        """
        pass

    def name(self):
        return str(self.__class__.__name__)

    def _initialize_logger(self, exp_dir):
        """Initialize logger of optimizer

        :param exp_dir: path of experiment directory
        :rtype exp_dir: str
        """

        # configure logger
        self.log_file = exp_dir + "/optimizer.log"
        if not hdfs.exists(self.log_file):
            hdfs.dump("", self.log_file)
        self.fd = hdfs.open_file(self.log_file, flags="w")
        self._log("Initialized Optimizer Logger")

    def _initialize(self, exp_dir):
        """
        initialize the optimizer and configure logger.

        :param exp_dir: path of experiment directory
        :rtype exp_dir: str
        """
        # init logger of optimizer
        self._initialize_logger(exp_dir=exp_dir)
        # optimizer intitialization routine
        self.initialize()
        self._log("Initilized Optimizer {}: \n {}".format(self.name(), self.__dict__))
        # init logger of pruner
        if self.pruner:
            self.pruner.initialize_logger(exp_dir=exp_dir)

    def _log(self, msg):
        if self.fd and not self.fd.closed:
            msg = datetime.now().isoformat() + ": " + str(msg)
            self.fd.write((msg + "\n").encode())

    def _close_log(self):
        if not self.fd.closed:
            self.fd.flush()
            self.fd.close()
