#
#   Copyright 2021 Logical Clocks AB
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

from __future__ import annotations

from typing import Union

from maggy import Searchspace
from maggy.earlystop import AbstractEarlyStop
from maggy.optimizer import AbstractOptimizer
from maggy.experiment_config import LagomConfig


class OptimizationConfig(LagomConfig):
    """Config class for hyperparameter optimization experiments."""

    def __init__(
        self,
        num_trials: int,
        optimizer: Union[str, AbstractOptimizer],
        searchspace: Searchspace,
        optimization_key: str = "metric",
        direction: str = "max",
        es_interval: int = 1,
        es_min: int = 10,
        es_policy: Union[str, AbstractEarlyStop] = "median",
        name: str = "HPOptimization",
        description: str = "",
        hb_interval: int = 1,
    ):
        """Initializes HP optimization experiment parameters.

        :param num_trials: Controls how many seperate runs are conducted during the hp search.
        :param optimizer: Optimizer type for searching the hp searchspace.
        :param searchspace: A Searchspace object configuring the names, types and ranges of hps.
        :param optimization_key: Name of the metric to use for hp search evaluation.
        :param direction: Direction of optimization.
        :param es_interval: Early stopping polling frequency during an experiment run.
        :param es_min: Minimum number of experiments to conduct before starting the early stopping
            mechanism. Useful to establish a baseline for performance estimates.
        :param es_policy: Early stopping policy which formulates a rule for triggering aborts.
        :param name: Experiment name.
        :param description: A description of the experiment.
        :param hb_interval: Heartbeat interval with which the server is polling.
        """
        super().__init__(name, description, hb_interval)
        if not num_trials > 0:
            raise ValueError("Number of trials should be greater than zero!")
        self.num_trials = num_trials
        self.optimizer = optimizer
        self.optimization_key = optimization_key
        self.searchspace = searchspace
        self.direction = direction
        self.es_policy = es_policy
        self.es_interval = es_interval
        self.es_min = es_min
