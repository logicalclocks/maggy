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

import os

import torch
from petastorm.reader import make_reader, make_batch_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader

from maggy.core.environment.singleton import EnvSing


def MaggyDataLoader(dataset, *args, **kwargs):
    """Factory function for Maggy data loaders.

    Breach of naming convention intentional to emphasize the fact that the factory returns a
    DataLoader object.
    """
    if isinstance(dataset, torch.utils.data.Dataset) or dataset is None:
        return MaggyTorchDataLoader(dataset, *args, **kwargs)
    return MaggyPetaDataLoader(dataset, *args, **kwargs)


class MaggyTorchDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        **_,
    ):
        num_workers = int(os.environ["WORLD_SIZE"])  # Is set at lagom startup.
        sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset)
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            sampler,
            batch_sampler,
            num_workers,
            collate_fn,
            pin_memory,
            drop_last,
            timeout,
            worker_init_fn,
            multiprocessing_context,
            generator,
        )
        self.iterator = None

    def __iter__(self):
        self.iterator = (
            super().__iter__()
        )  # Reload the dataset when new iterator requested.
        return self

    def __next__(self):
        data = self.iterator.__next__()
        return to_cuda(data)


class MaggyPetaDataLoader(PetastormDataLoader):
    def __init__(self, dataset, batch_size=1, **_):
        num_workers = int(os.environ["WORLD_SIZE"])  # Is set at lagom startup.
        rank = int(os.environ["RANK"])
        is_peta_ds = EnvSing.get_instance().exists(
            dataset.rstrip("/") + "/_common_metadata"
        )
        # Make reader only compatible with petastorm dataset.
        ds_type = "Petastorm" if is_peta_ds else "Parquet"
        print(f"{ds_type} dataset detected in folder {dataset}")
        reader_factory = make_reader if is_peta_ds else make_batch_reader
        reader = reader_factory(dataset, cur_shard=rank, shard_count=num_workers)
        super().__init__(reader, batch_size=batch_size)
        self.iterator = None

    def __iter__(self):
        self.iterator = (
            super().__iter__()
        )  # Reload the dataset when new iterator requested.
        return self

    def __next__(self):
        data = self.iterator.__next__()
        return to_cuda(data)


def to_cuda(data):
    """Recurses into data, transfers tensors to GPU."""
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_cuda(data[key])
        return data
    if isinstance(data, list):
        for idx, _ in enumerate(data):
            temp = to_cuda(data[idx])
            data[idx] = temp
        return data
    if isinstance(data, torch.Tensor):
        return data.cuda()
    raise ValueError(f"Type {type(data)} currently not supported!")
