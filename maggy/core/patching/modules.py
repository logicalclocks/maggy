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

try:
    from deepspeed.pipe import PipelineModule
    from deepspeed.runtime.engine import DeepSpeedEngine
    from fairscale.nn import (
        FullyShardedDataParallel as FairscaleFullyShardedDataParallel,
    )
except ImportError:
    print(
        """Warning: deepspeed and/or fairscale import failed. DeepSpeed backend and zero_lvl 3
          won't be available"""
    )


def get_maggy_ddp_wrapper(module: Type[TorchModule]):
    """Factory function for MaggyDDPModuleWrapper.

    :param module: PyTorch module passed by the user.
    """

    class MaggyDDPModuleWrapper(TorchDistributedDataParallel):
        """Wrapper around PyTorch's DDP Module.

        The wrapper replaces the user's module. Since the module's signature needs to be preserved,
        we cannot add the module as an additional parameter during initialization. Instead, it is
        configured by its factory function.
        """

        __module = module  # Avoid overwriting torch module

        def __init__(self, *args: Any, **kwargs: Any):
            """Initializes the previously set module, moves it to the GPU and initializes a DDP
            module with it.

            :param args: Arguments passed by the user for module initialization.
            :param kwargs: Keyword arguments passed by the user for module initialization.
            """
            # Avoid self because bound method adds to args which makes the function call fail
            model = MaggyDDPModuleWrapper.__module(*args, **kwargs).cuda()
            super().__init__(model)

    return MaggyDDPModuleWrapper


def get_maggy_fairscale_wrapper(module: TorchModule, mixed_precision: bool):
    """Factory function for MaggyFairScaleModuleWrapper.

    :param module: PyTorch module passed by the user.
    :param mixed_precision: Switches on mixed precision for the FairscaleModule.
    """

    class MaggyFairScaleModuleWrapper(FairscaleFullyShardedDataParallel):
        """Wrapper around Fairscale's FullyShardedDataParallel Module.

        The wrapper replaces the user's module. Since the module's signature needs to be preserved,
        we cannot add the module as an additional parameter during initialization. Instead, it is
        configured by its factory function.
        """

        __module = module
        __mixed_precision = mixed_precision

        def __init__(self, *args: Any, **kwargs: Any):
            """Initializes the previously set module, moves it to the GPU and initializes a
            Fairscale FullyShardedDataParallel module with it.

            :param args: Arguments passed by the user for module initialization.
            :param kwargs: Keyword arguments passed by the user for module initialization.
            """
            # Avoid self because bound method adds to args which makes the function call fail
            model = MaggyFairScaleModuleWrapper.__module(*args, **kwargs).cuda()
            super().__init__(model, mixed_precision=self.__mixed_precision)

    return MaggyFairScaleModuleWrapper


def get_maggy_deepspeed_wrapper(module: TorchModule, config_params: dict):
    """Factory function for MaggyDeepSpeedModuleWrapper.

    :param module: PyTorch module passed by the user.
    :param mixed_precision: DeepSpeed config dict passed by the user.
    """
    assert (
        module != PipelineModule
    ), """Maggy currently doesn't support pipeline
             modules with DeepSpeed ZeRO."""

    class MaggyDeepSpeedModuleWrapper(DeepSpeedEngine):
        """Wrapper around DeepSpeed's DeepSpeedEngine.

        The wrapper replaces the user's module. Since the module's signature needs to be preserved,
        we cannot add the module as an additional parameter during initialization. Instead, it is
        configured by its factory function.
        """

        __module = module
        __config_params = config_params

        def __init__(self, *args, **kwargs):
            """Initializes the previously set module and initializes a DeepSpeedEngine with it.

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
                config_params=self.__config_params,
            )

    return MaggyDeepSpeedModuleWrapper
