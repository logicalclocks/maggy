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

import typing
from typing import Union, Type, Optional
from abc import ABC

from maggy import Searchspace
from maggy.earlystop import AbstractEarlyStop
from maggy.optimizer import AbstractOptimizer
from maggy.ablation.ablationstudy import AblationStudy
from maggy.ablation.ablator import AbstractAblator

if typing.TYPE_CHECKING:
    import torch


class LagomConfig(ABC):
    """Base class for lagom configuration classes.
    """

    def __init__(self, name: str, description: str, hb_interval: int):
        """Initializes basic experiment info.

        :param name: Experiment name.
        :param description: A description of the experiment.
        :param hb_interval: Heartbeat interval with which the server is polling.
        """
        self.name = name
        self.description = description
        self.hb_interval = hb_interval


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


class AblationConfig(LagomConfig):
    """Config class for ablation study experiments."""

    def __init__(
        self,
        ablation_study: AblationStudy,
        ablator: Union[str, AbstractAblator] = "loco",
        direction: str = "max",
        name: str = "ablationStudy",
        description: str = "",
        hb_interval: int = 1,
    ):
        """Initializes ablation study experiment parameters.

        :param ablation_study: Ablation study object that defines the entry point into the
            experiment.
        :param ablator: An instance of `AbstractAblator` or a supported ablator name that controls
            the manner in which parts of the model are ablated.
        :param direction: Optimization direction to evaluate the experiments.
        :param name: Experiment name.
        :param description: A description of the experiment.
        :param hb_interval: Heartbeat interval with which the server is polling.
        """
        super().__init__(name, description, hb_interval)
        self.ablator = ablator
        self.ablation_study = ablation_study
        self.direction = direction


class TorchDistributedConfig(LagomConfig):
    """Config class for running distributed PyTorch training."""

    BACKENDS = ["ddp", "deepspeed"]

    def __init__(
        self,
        module: Type[torch.nn.Module],
        train_set: Optional[Union[str, torch.util.data.Dataset]] = None,
        test_set: Optional[Union[str, torch.util.data.Dataset]] = None,
        hparams: dict = None,
        backend: str = "ddp",
        ddp3_mp: bool = False,
        zero_lvl: int = 0,
        deepspeed_config: dict = None,
        name: str = "torchDist",
        hb_interval: int = 1,
        description: str = "",
    ):
        """Initializes PyTorch distributed training parameters.

        :param module: A PyTorch module class. Note that this has to be the class itself, not
            an instance.
        :param train_set: The training set for the training function. If you want to load the set
            inside the training function, this can be disregarded.
        :param test_set: The test set for the training function. If you want to load the set
            inside the training function, this can be disregarded.
        :param hparams: Hyperparameters that should be used during model initialization. Primarily
            used to give an interface for hp optimization.
        :param backend: The backend engine used for training. Note that `deepspeed` needs syntax
            changes to a normal PyTorch script!
        :param ddp3_mp: Used to control the use of mixed precision training in `ddp` backend mode
            with model sharding (`zero_lvl` 3).
        :param zero_lvl: Sets the ZeRO optimization stages for `ddp`. Note: When using `deepspeed`
            backend, overwrites `deepspeed_config` zero level!
        :param deepspeed_config: A dictionary that represents a valid deepspeed ZeRO optimizer
            config. For information on the config, see https://www.deepspeed.ai/docs/config-json/.
        :param name: Experiment name.
        :param hb_interval: A description of the experiment.
        :param description: Heartbeat interval with which the server is polling.
        """
        super().__init__(name, description, hb_interval)
        self.module = module
        self.train_set = train_set
        self.test_set = test_set
        if backend not in self.BACKENDS:
            raise ValueError(
                """Backend {} not supported by Maggy.
                 Supported types are: {}""".format(
                    backend, self.BACKENDS
                )
            )
        self.backend = backend
        self.ddp3_mp = ddp3_mp
        self.hparams = hparams if hparams else {}
        self.zero_lvl = zero_lvl
        self.ds_config = deepspeed_config
