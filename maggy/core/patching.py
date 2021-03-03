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
from torch.utils.data import DataLoader, Dataset
from petastorm.reader import make_reader, make_batch_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader

from maggy.core.environment.singleton import EnvSing


class MaggyDataLoader(DataLoader, PetastormDataLoader):

    loaders = {"pytorch": DataLoader, "petastorm": PetastormDataLoader}

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
        # Distinguish between Torch dataloader and Petastorm dataloader
        # Super init avoided to make sure MaggyDataLoader inherits correctly based on dataset type.
        if isinstance(dataset, Dataset):
            sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset)
            DataLoader.__init__(
                self,
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
            reader = reader_factory(dataset, cur_shard=rank, shard_count=num_workers)
            PetastormDataLoader.__init__(self, reader, batch_size=batch_size)
            self.mode = "petastorm"
        self.iterator = None

    def __iter__(self):
        # Reload the dataset when new iterator requested.
        self.iterator = self.loaders[self.mode].__iter__(self)
        return self

    def __next__(self):
        data = self.iterator.__next__()
        return _to_cuda(data)


def _to_cuda(data):
    """Recurses into data, transfers tensors to GPU."""
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = _to_cuda(data[key])
        return data
    if isinstance(data, list):
        for idx, _ in enumerate(data):
            temp = _to_cuda(data[idx])
            data[idx] = temp
        return data
    if isinstance(data, torch.Tensor):
        return data.cuda()
    raise ValueError(f"Type {type(data)} currently not supported!")
