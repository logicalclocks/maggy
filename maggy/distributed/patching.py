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
import petastorm
from petastorm.pytorch import DataLoader as PetastormDataLoader


class MaggyDataLoader(torch.utils.data.DataLoader):
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
        *,
        prefetch_factor=2,
        persistent_workers=False
    ):
        num_workers = int(
            os.environ["WORLD_SIZE"]
        )  # Expected to be set by maggy startup.
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
        for idx, _ in enumerate(
            data
        ):  # Explicit use of index to avoid simply making copies.
            data[idx] = data[
                idx
            ].cuda()  # Distributed training requires cuda to be available.
        return data


class MaggyPetaDataLoader(PetastormDataLoader):
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
        *,
        prefetch_factor=2,
        persistent_workers=False
    ):
        num_workers = int(
            os.environ["WORLD_SIZE"]
        )  # Expected to be set by maggy startup.
        rank = int(os.environ["RANK"])
        batch_reader = petastorm.reader.make_batch_reader(
            dataset, cur_shard=rank, shard_count=num_workers
        )
        super().__init__(batch_reader, batch_size=batch_size)
        self.iterator = None

    def __iter__(self):
        self.iterator = (
            super().__iter__()
        )  # Reload the dataset when new iterator requested.
        return self

    def __next__(self):
        data = self.iterator.__next__()
        for key in data.keys():
            data[key] = data[key].cuda()  # Distributed training requires cuda.
        return data
