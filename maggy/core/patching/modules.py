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

from types import SimpleNamespace
from typing import Type, Any

from torch.nn import Module as TorchModule
from torch.nn.parallel import DistributedDataParallel as TorchDistributedDataParallel

from deepspeed.pipe import PipelineModule
from deepspeed.runtime.engine import DeepSpeedEngine
from fairscale.nn import FullyShardedDataParallel as FairscaleFullyShardedDataParallel


class MaggyDDPModuleWrapper(TorchDistributedDataParallel):
    """Wrapper around PyTorch's DDP Module.

    The wrapper replaces the user's module. Since the module's signature needs to be preserved, we
    cannot add the module as an additional parameter during initialization. Instead, the module is
    configured before initialization of the class by a user. Its class property `__module` is set
    by the executor function during patching. Note that this property has to be set appropriately
    for the wrapper to function properly.
    """

    __module = None  # Avoid overwriting torch module

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the previously set module, moves it to the GPU and initializes a DDP module
        with it.

        Note that the `__module` attribute has to be set BEFORE the constructor is called for the
        first time!

        :param args: Arguments passed by the user for module initialization.
        :param kwargs: Keyword arguments passed by the user for module initialization.
        """
        # Avoid self because bound method adds to args which makes the function call fail
        model = MaggyDDPModuleWrapper.__module(*args, **kwargs).cuda()
        super().__init__(model)

    @classmethod
    def config(cls, module: Type[TorchModule]) -> Type[MaggyDDPModuleWrapper]:
        """Sets the wrapper module class property.

        :param module: The PyTorch module that is to be wrapped.

        :returns: A MaggyDDPModuleWrapper class type with the registered properties.
        """
        cls.__module = module
        return cls


class MaggyFairScaleModuleWrapper(FairscaleFullyShardedDataParallel):
    """Wrapper around Fairscale's FullyShardedDataParallel Module.

    The wrapper replaces the user's module. Since the module's signature needs to be preserved, we
    cannot add the module as an additional parameter during initialization. Instead, the module is
    configured before initialization of the class by a user. Its class properties `__module` and
    `mixed_precision` are set by the executor function during patching. Note that these properties
    have to be set appropriately for the wrapper to function properly.
    """

    __module = None
    __mixed_precision = False

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes the previously set module, moves it to the GPU and initializes a Fairscale
        FullyShardedDataParallel module with it.

        Note that the `__module` attribute has to be set BEFORE the constructor is called for the
        first time!

        :param args: Arguments passed by the user for module initialization.
        :param kwargs: Keyword arguments passed by the user for module initialization.
        """
        # Avoid self because bound method adds to args which makes the function call fail
        model = MaggyFairScaleModuleWrapper.__module(*args, **kwargs).cuda()
        super().__init__(model, mixed_precision=self.__mixed_precision)

    @classmethod
    def config(
        cls, module: Type[TorchModule], mixed_precision: bool
    ) -> Type[MaggyFairScaleModuleWrapper]:
        """Sets the wrapper module class and mixed_precision properties.

        :param module: The PyTorch module that is to be wrapped.
        :param mixed_precision: Determines if mixed precision is used for the model's parameters
        during training for increased speed.

        :returns: A MaggyFairScaleModuleWrapper class type with the registered properties.
        """
        cls.__module = module
        cls.__mixed_precision = mixed_precision
        return cls


class MaggyDeepSpeedModuleWrapper(DeepSpeedEngine):
    """Wrapper around DeepSpeed's DeepSpeedEngine.

    The wrapper replaces the user's module. Since the module's signature needs to be preserved, we
    cannot add the module as an additional parameter during initialization. Instead, the module is
    configured before initialization of the class by a user. Its class properties `__module` and
    `config_params` are set by the executor function during patching. Note that these properties
    have to be set appropriately for the wrapper to function properly.
    """

    __module = None
    config_params = None

    def __init__(self, *args, **kwargs):
        """Initializes the previously set module and initializes a DeepSpeedEngine with it.

        Note that the `__module` attribute has to be set BEFORE the constructor is called for the
        first time!

        :param args: Arguments passed by the user for module initialization.
        :param kwargs: Keyword arguments passed by the user for module initialization.
        """
        # Avoid self because bound method adds to args which makes the function call fail.
        # No .cuda() calls for DeepSpeed necessary.
        model = MaggyDeepSpeedModuleWrapper.__module(*args, **kwargs)
        ds_args = SimpleNamespace(local_rank=0)
        super().__init__(
            ds_args,
            model,
            model_parameters=model.parameters(),
            config_params=self.config_params,
        )

    @classmethod
    def config(
        cls, module: Type[TorchModule], config_params: dict
    ) -> Type[MaggyDeepSpeedModuleWrapper]:
        """Sets the wrapper module class and config parameters properties.

        :param module: The PyTorch module that is to be wrapped.
        :param config_params: Configuration dictionary for the DeepSpeedEngine.

        :returns: A MaggyDeepSpeedModuleWrapper class type with the registered properties.
        """
        assert (
            module != PipelineModule
        ), """Maggy currently doesn't support pipeline
             modules with DeepSpeed ZeRO."""
        cls.__module = module
        cls.config_params = config_params
        return cls
