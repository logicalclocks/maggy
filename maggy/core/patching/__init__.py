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

import torch

from .dataloader import MaggyDataLoader, MaggyPetastormDataLoader
from .modules import (
    get_maggy_ddp_wrapper,
    get_maggy_fairscale_wrapper,
    get_maggy_deepspeed_wrapper,
)

__all__ = [
    "get_maggy_ddp_wrapper",
    "get_maggy_fairscale_wrapper",
    "get_maggy_deepspeed_wrapper",
    "MaggyDataLoader",
    "MaggyPetastormDataLoader",
]

# Check torch version, only import ZeroRedundancyOptimizer if >= 1.8
_torch_version = torch.__version__.split(".")
if int(_torch_version[0]) > 1 or int(_torch_version[1]) >= 8:
    from .optim import (
        MaggyZeroAdadelta,
        MaggyZeroAdagrad,
        MaggyZeroAdam,
        MaggyZeroAdamW,
        MaggyZeroSparseAdam,
        MaggyZeroAdamax,
        MaggyZeroASGD,
        MaggyZeroLBFGS,
        MaggyZeroRMSprop,
        MaggyZeroRprop,
        MaggyZeroSGD,
    )

    __all__ += [
        "MaggyZeroAdadelta",
        "MaggyZeroAdagrad",
        "MaggyZeroAdam",
        "MaggyZeroAdamW",
        "MaggyZeroSparseAdam",
        "MaggyZeroAdamax",
        "MaggyZeroASGD",
        "MaggyZeroLBFGS",
        "MaggyZeroRMSprop",
        "MaggyZeroRprop",
        "MaggyZeroSGD",
    ]
