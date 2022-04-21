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

import inspect
from typing import Any
from abc import ABC, abstractclassmethod


import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer


class MaggyZeroOptimizer(ZeroRedundancyOptimizer, ABC):
    """Abstract base class for Maggy's optimizer patching classes."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes a ZeroRedundancyOptimizer with the defined optim_cls as optimizer class.

        Passes any arguments for initialization of the default optimizer to the Zero optimizer.
        :param args: Optimizer args. Get reassigned into kwargs.
        :param kwargs: Optimizer kwargs.
        """
        # Move args to kwargs to pass args into kwargs only ZeroRedundancyOptimizer
        arg_spec = inspect.getfullargspec(self.optim_cls.__init__)
        for idx, arg in enumerate(args):
            kwargs[arg_spec.args[idx + 1]] = arg  # +1 to skip self in arg_spec
        params = kwargs.pop("params", None)
        super().__init__(
            params, self.optim_cls, group=None, bucket_cap_kb=2**24, **kwargs
        )

    @property
    @abstractclassmethod
    def optim_cls(cls: optim.Optimizer) -> MaggyZeroOptimizer:
        """Optimizer class property needs to be defined by each implementation of the base class."""
        raise NotImplementedError


class MaggyZeroAdadelta(MaggyZeroOptimizer):
    """Maggy's Zero wrapper around torch's Adadelta optimizer."""

    optim_cls = optim.Adadelta


class MaggyZeroAdagrad(MaggyZeroOptimizer):
    """Maggy's Zero wrapper around torch's Adagrad optimizer."""

    optim_cls = optim.Adagrad


class MaggyZeroAdam(MaggyZeroOptimizer):
    """Maggy's Zero wrapper around torch's Adam optimizer."""

    optim_cls = optim.Adam


class MaggyZeroAdamW(MaggyZeroOptimizer):
    """Maggy's Zero wrapper around torch's AdamW optimizer."""

    optim_cls = optim.AdamW


class MaggyZeroSparseAdam(MaggyZeroOptimizer):
    """Maggy's Zero wrapper around torch's SparseAdam optimizer."""

    optim_cls = optim.SparseAdam


class MaggyZeroAdamax(MaggyZeroOptimizer):
    """Maggy's Zero wrapper around torch's Adamax optimizer."""

    optim_cls = optim.Adamax


class MaggyZeroASGD(MaggyZeroOptimizer):
    """Maggy's Zero wrapper around torch's ASGD optimizer."""

    optim_cls = optim.ASGD


class MaggyZeroLBFGS(MaggyZeroOptimizer):
    """Maggy's Zero wrapper around torch's LBFGS optimizer."""

    optim_cls = optim.LBFGS


class MaggyZeroRMSprop(MaggyZeroOptimizer):
    """Maggy's Zero wrapper around torch's RMSprop optimizer."""

    optim_cls = optim.RMSprop


class MaggyZeroRprop(MaggyZeroOptimizer):
    """Maggy's Zero wrapper around torch's Rprop optimizer."""

    optim_cls = optim.Rprop


class MaggyZeroSGD(MaggyZeroOptimizer):
    """Maggy's Zero wrapper around torch's SGD optimizer."""

    optim_cls = optim.SGD
