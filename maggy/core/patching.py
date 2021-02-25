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
from pytorch_lightning import Trainer

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


class MaggyTrainer(Trainer):
    def __init__(
        self,
        logger=True,
        checkpoint_callback=True,
        callbacks=None,
        default_root_dir=None,
        gradient_clip_val=0,
        process_position=0,
        num_nodes=1,
        num_processes=1,
        gpus=None,
        auto_select_gpus=False,
        tpu_cores=None,
        log_gpu_memory=None,
        progress_bar_refresh_rate=None,
        overfit_batches=0.0,
        track_grad_norm=-1,
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        accumulate_grad_batches=1,
        max_epochs=None,
        min_epochs=None,
        max_steps=None,
        min_steps=None,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        limit_predict_batches=1.0,
        val_check_interval=1.0,
        flush_logs_every_n_steps=100,
        log_every_n_steps=50,
        accelerator=None,
        sync_batchnorm=False,
        precision=32,
        weights_summary="top",
        weights_save_path=None,
        num_sanity_val_steps=2,
        truncated_bptt_steps=None,
        resume_from_checkpoint=None,
        profiler=None,
        benchmark=False,
        deterministic=False,
        reload_dataloaders_every_epoch=False,
        auto_lr_find=False,
        replace_sampler_ddp=True,
        terminate_on_nan=False,
        auto_scale_batch_size=False,
        prepare_data_per_node=True,
        plugins=None,
        amp_backend="native",
        amp_level="O2",
        distributed_backend=None,
        move_metrics_to_cpu=False,
        multiple_trainloader_mode="max_size_cycle",
        stochastic_weight_avg=False,
    ):
        gpus = 1  # Each Spark worker has only 1 GPU assigned.
        num_nodes = int(os.environ["WORLD_SIZE"])
        distributed_backend = "ddp"
        super().__init__(
            logger,
            checkpoint_callback,
            callbacks,
            default_root_dir,
            gradient_clip_val,
            process_position,
            num_nodes,
            num_processes,
            gpus,
            auto_select_gpus,
            tpu_cores,
            log_gpu_memory,
            progress_bar_refresh_rate,
            overfit_batches,
            track_grad_norm,
            check_val_every_n_epoch,
            fast_dev_run,
            accumulate_grad_batches,
            max_epochs,
            min_epochs,
            max_steps,
            min_steps,
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            limit_predict_batches,
            val_check_interval,
            flush_logs_every_n_steps,
            log_every_n_steps,
            accelerator,
            sync_batchnorm,
            precision,
            weights_summary,
            weights_save_path,
            num_sanity_val_steps,
            truncated_bptt_steps,
            resume_from_checkpoint,
            profiler,
            benchmark,
            deterministic,
            reload_dataloaders_every_epoch,
            auto_lr_find,
            replace_sampler_ddp,
            terminate_on_nan,
            auto_scale_batch_size,
            prepare_data_per_node,
            plugins,
            amp_backend,
            amp_level,
            distributed_backend,
            move_metrics_to_cpu,
            multiple_trainloader_mode,
            stochastic_weight_avg,
        )
