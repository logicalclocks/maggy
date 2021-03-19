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

import os
from typing import Type, Union, Optional, Any, Callable
import collections

import torch
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DataLoader as TorchDataLoader
from petastorm.reader import make_reader, make_batch_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader
from petastorm.transform import TransformSpec

from maggy.core.environment.singleton import EnvSing


class MaggyDataLoader(TorchDataLoader, PetastormDataLoader):
    """Monkey patching class for PyTorch's DataLoader.

    Patches the DataLoader to include a distributed sampler. In case a path
    is provided instead of a Dataset, assumes a Parquet dataset in that folder
    and initializes a Petastorm DataLoader instead (which in turn builds on
    PyTorch's DataLoader). Uses environment variables for infos such as world
    size for both DataLoader and PetastormDataLoader. These can assumed to be
    present since Maggy's distributed experiment sets them prior to running the
    training.
    Automatically moves training data to the GPU since distributed training
    requires execution on GPUs.
    """

    loaders = {"pytorch": TorchDataLoader, "petastorm": PetastormDataLoader}

    def __init__(
        self,
        dataset: Union[Type[Dataset], str],
        batch_size: int = 1,
        shuffle: Any = False,
        sampler: Optional[Sampler[int]] = None,
        batch_sampler: Optional[Any] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        transform_spec: Optional[TransformSpec] = None,  # Petastorm option
        **_: Any,
    ):
        """Initializes either a torch DataLoader or a Petastorm DataLoader.

        Decides which one to choose based on dataset's type.

        :param dataset: Either a PyTorch Dataset or a path to a Parquet dataset.
        :param batch_size: How many samples per batch to load (default: ``1``).
        :param shuffle: Discarded, not compatible with Maggy.
        :param sampler: Discarded, gets replaced by DistributedSampler.
        :param batch_sampler: Discarded, not compatible with Maggy.
        :param num_workers: Number of subprocesses for data loading.
        :param collate_fn: Merges a list of samples to a minibatch of tensors.
        :param pin_memory: Automatically transfer tensors to GPU.
        :param drop_last: Drop last incomplete batch.
        :param timeout: Timeout for collecting a batch.
        :param worker_init_fn: Executed on each worker with worker ID.
        :param _: Argument catch to stay compatible with PyTorch.
        """
        # Distinguish between Torch dataloader and Petastorm dataloader
        # Super init avoided to make sure MaggyDataLoader inherits correctly based on dataset type.
        if isinstance(dataset, Dataset):
            sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset)
            TorchDataLoader.__init__(
                self,
                dataset,
                batch_size,
                shuffle=False,
                sampler=sampler,
                batch_sampler=None,
                num_workers=0,  # Multiprocessing workers do not work at the moment.
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                timeout=timeout,
                worker_init_fn=worker_init_fn,
            )
            self.mode = "pytorch"
        else:
            num_workers = int(os.environ["WORLD_SIZE"])  # Is set at lagom startup.
            rank = int(os.environ["RANK"])
            is_peta_ds = EnvSing.get_instance().exists(
                dataset.rstrip("/") + "/_common_metadata"
            )
            # Make reader only compatible with petastorm dataset.
            ds_type = "Petastorm" if is_peta_ds else "Parquet"
            print(f"{ds_type} dataset detected in folder {dataset}")
            reader_factory = make_reader if is_peta_ds else make_batch_reader
            reader = reader_factory(
                dataset,
                cur_shard=rank,
                shard_count=num_workers,
                transform_spec=TransformSpec(transform_spec),
            )
            PetastormDataLoader.__init__(self, reader, batch_size=batch_size)
            self.mode = "petastorm"
        self.iterator = None

    def __iter__(self) -> MaggyDataLoader:
        # Reload the dataset when new iterator requested.
        self.iterator = self.loaders[self.mode].__iter__(self)
        return self

    def __next__(self) -> Union[torch.Tensor, list, dict]:
        data = self.iterator.__next__()
        return self._to_cuda(data)

    def __len__(self):
        if self.mode == "petastorm":
            raise TypeError("Petastorm dataloader does not support __len__.")
        return super().__len__()

    @classmethod
    def _to_cuda(
        cls, data: Union[torch.Tensor, list, dict]
    ) -> Union[torch.Tensor, list, dict]:
        """Recurses into data, transfers tensors to GPU.

        :param data: The data structure to be transferred.

        :raises TypeError: In case of unsupported data structures.

        :returns: The transfered data structure.
        """
        if isinstance(data, collections.abc.Mapping):
            return {key: cls._to_cuda(val) for key, val in data.items()}
        if isinstance(data, (list, tuple)):
            data_list = [cls._to_cuda(el) for el in data]
            return data_list if isinstance(data, list) else tuple(data_list)
        if isinstance(data, torch.Tensor):
            return data.cuda()
        raise TypeError(f"Type {type(data)} currently not supported!")
