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
from typing import Union, Type, Optional, List
from maggy.config import LagomConfig
from maggy.core import config as mc

if typing.TYPE_CHECKING:
    import torch


class TorchDistributedConfig(LagomConfig):
    """LagomConfig class for running distributed PyTorch training."""

    BACKENDS = ["torch", "deepspeed"]

    def __init__(
        self,
        module: Union[Type[torch.nn.Module], List[Type[torch.nn.Module]]],
        dataset: List[Optional[Union[str, torch.util.data.Dataset]]] = None,
        hparams: dict = None,
        backend: str = "torch",
        mixed_precision: bool = False,
        zero_lvl: int = 0,
        deepspeed_config: dict = None,
        name: str = "torchDist",
        hb_interval: int = 1,
        description: str = "",
    ):
        """Initializes PyTorch distributed training parameters.

        :param module: A PyTorch module class or list of PyTorch module classes.
            Note that this has to be the class itself, not an instance.
        :param dataset: A List of strings containing the dataset path or list of torch.util.data.Dataset.
        these datasets represent the ones you are going to use in the training function. For example,
        if you have 2 datasets for training and testing, pass an array with [train_set, test_set] and extract them in
        the training function. If you want to load the set inside the training function, this can be disregarded.
        :param hparams: Hyperparameters that should be used during model initialization. Primarily
            used to give an interface for hp optimization.
        :param backend: The backend framework used for training. Note that `deepspeed` needs syntax
            changes to a normal PyTorch script!
        :param mixed_precision: Used to control the use of mixed precision training in `torch`
            backend mode with model sharding (`zero_lvl` 3).
        :param zero_lvl: Sets the ZeRO optimization stages for `torch`. Note: When using `deepspeed`
            backend, overwrites `deepspeed_config` zero level!
        :param deepspeed_config: A dictionary that represents a valid deepspeed ZeRO optimizer
            config. For information on the config, see https://www.deepspeed.ai/docs/config-json/.
        :param name: Experiment name.
        :param hb_interval: Heartbeat interval with which the server is polling.
        :param description: A description of the experiment.
        """
        super().__init__(name, description, hb_interval)
        mc.initialize()
        if not mc.is_spark_available():
            raise NotImplementedError(
                "Torch Distributed Training can run only on a Spark kernel."
            )
        self.module = module
        self.dataset = dataset
        if backend not in self.BACKENDS:
            raise ValueError(
                """Backend {} not supported by Maggy.
                 Supported types are: {}""".format(
                    backend, self.BACKENDS
                )
            )
        self.backend = backend
        self.mixed_precision = mixed_precision
        self.hparams = hparams if hparams else {}
        self.zero_lvl = zero_lvl
        self.ds_config = deepspeed_config
