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

import builtins as __builtin__
import inspect
import traceback
import os
import datetime
import random
import socket
from typing import Callable, Union, Any, Tuple, Type, List

import torch
import torch.distributed as dist

import numpy as np
import deepspeed

from maggy import util
from maggy import tensorboard
from maggy.config import TorchDistributedConfig
from maggy.core.rpc import Client
from maggy.core.reporter import Reporter
from maggy.core.patching import MaggyDataLoader
from maggy.core.patching import (
    get_maggy_ddp_wrapper,
    get_maggy_fairscale_wrapper,
    get_maggy_deepspeed_wrapper,
)

from maggy.core.environment.singleton import EnvSing

_torch_version = torch.__version__.split(".")  # Check compatibility with 1.8
if int(_torch_version[0]) > 1 or int(_torch_version[1]) >= 8:
    from maggy.core.patching import (
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


def torch_dist_executor_fn(
    train_fn: Callable,
    config: TorchDistributedConfig,
    app_id: int,
    run_id: int,
    server_addr: str,
    hb_interval: int,
    secret: str,
    log_dir: str,
) -> Callable:
    """Wraps the user supplied training function in order to be passed to the Spark Executors.

    :param train_fn: Original training function.
    :param config: Experiment config.
    :param app_id: Maggy application ID.
    :param run_id: Maggy run ID.
    :param server_addr: IP of the Maggy worker registration RPC server.
    :param hb_interval: Worker heartbeat interval.
    :param secret: Secret string to authenticate messages.
    :param log_dir: Location of the logger file directory on the file system.

    :returns: Patched function to execute on the Spark executors.
    """

    def wrapper_function(_: Any) -> None:
        """Patched function from dist_executor_fn factory.

        :param _: Necessary catch for the iterator given by Spark to the
        function upon foreach calls. Can safely be disregarded.
        """
        EnvSing.get_instance().set_ml_id(app_id, run_id)

        partition_id, _ = util.get_partition_attempt_id()

        client = EnvSing.get_instance().get_client(
            server_addr,
            partition_id,
            hb_interval,
            secret,
            socket.socket(socket.AF_INET, socket.SOCK_STREAM),
        )
        log_file = log_dir + "/executor_" + str(partition_id) + ".log"

        builtin_print = __builtin__.print
        reporter = Reporter(log_file, partition_id, 0, builtin_print)

        def maggy_print(*args, **kwargs):
            builtin_print(*args, **kwargs)
            reporter.log(" ".join(str(x) for x in args), True)

        __builtin__.print = maggy_print

        try:
            _register_with_servers(client, reporter, partition_id)
            tb_logdir, trial_log_file = _setup_logging(reporter, log_dir)
            reporter.log("Awaiting worker reservations.", True)
            client.await_reservations()
            reporter.log("Reservations complete, configuring PyTorch.", True)
            master_config = client.get_message("EXEC_CONFIG")[0]
            if not master_config:
                reporter.log("RuntimeError: PyTorch registration failed.", True)
                raise RuntimeError("PyTorch registration failed.")
            addr, port = master_config["host_port"].split(":")
            torch_config = {
                "MASTER_ADDR": addr,
                "MASTER_PORT": port,
                "WORLD_SIZE": str(master_config["num_executors"]),
                "RANK": str(partition_id),
                "LOCAL_RANK": str(0),  # DeepSpeed requires local rank.
                "NCCL_BLOCKING_WAIT": "1",
                "NCCL_DEBUG": "INFO",
            }
            tensorboard._register(tb_logdir)
            reporter.log(f"Torch config is {torch_config}", True)

            _setup_torch_env(torch_config)
            _sanitize_config(config)
            _init_cluster(timeout=60, random_seed=0)
            module = _wrap_module_dispatcher(config)
            _monkey_patch_pytorch(config.zero_lvl)

            reporter.log("Starting distributed training.", True)
            sig = inspect.signature(train_fn)

            kwargs = {}
            if sig.parameters.get("module", None):
                kwargs["module"] = module
            if sig.parameters.get("dataset", None):
                kwargs["dataset"] = config.dataset
            if sig.parameters.get("hparams", None):
                kwargs["hparams"] = config.hparams

            if sig.parameters.get("reporter", None):
                kwargs["reporter"] = reporter
                retval = train_fn(**kwargs)
            else:
                retval = train_fn(**kwargs)

            # todo: test this change
            retval_list = []
            if isinstance(retval, dict):
                for item in retval.items():
                    retval_list.append(item[1])
                retval = retval_list
            retval = util.handle_return_val(retval, tb_logdir, "Metric", trial_log_file)
            dist.barrier()  # Don't exit until all executors are done (else NCCL crashes)
            reporter.log("Finished distributed training.", True)
            client.finalize_metric(retval, reporter)
        except:  # noqa: E722
            reporter.log(traceback.format_exc())
            raise
        finally:
            reporter.close_logger()
            client.stop()
            client.close()

    return wrapper_function


def _register_with_servers(
    client: Client, reporter: Reporter, partition_id: int
) -> None:
    """Registers own address with server and starts heartbeat protocol.

    :param client: Client for communication with the server.
    :param reporter: Reporter responsible for heartbeat.
    :param partition_id: Executors partition ID from Sparks RDD.
    """
    client_addr = client.client_addr
    port = _get_open_port()
    host_port = client_addr[0] + ":" + str(port)
    exec_spec = {
        "partition_id": partition_id,
        "task_attempt": 0,
        "host_port": host_port,
        "trial_id": None,
    }
    client.register(exec_spec)
    client.start_heartbeat(reporter)


def _get_open_port() -> str:
    """Lets the OS choose a free socket and attempts to bind it.

    :returns: The port name.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))  # Bind to 0 lets OS choose a free socket.
    port = sock.getsockname()[1]
    sock.close()
    return port


def _setup_logging(reporter: Reporter, log_dir: str) -> Tuple[str, str]:
    """Sets up logging directories and files.

    :param reporter: Reporter responsible for logging.
    :param log_dir: Log directory path on the file system.

    :returns: Tuple containing the path of the tensorboard directory
        and the trial log file.
    """
    reporter.set_trial_id(0)
    tb_logdir = log_dir + "/" + "training_logs_" + str(reporter.partition_id)
    trial_log_file = tb_logdir + "/output.log"
    reporter.set_trial_id(0)
    # If trial is repeated, delete trial directory, except log file
    if EnvSing.get_instance().exists(tb_logdir):
        util.clean_dir(tb_logdir, [trial_log_file])
    else:
        EnvSing.get_instance().mkdir(tb_logdir)
    reporter.init_logger(trial_log_file)
    return tb_logdir, trial_log_file


def _setup_torch_env(torch_config: dict) -> None:
    """Registers the Torch config as environment variables on the worker.

    :param torch_config: Dictionary containing the values of the variables.
    """
    for env_variable in torch_config.keys():
        os.environ[env_variable] = str(torch_config[env_variable])


def _init_cluster(
    timeout: int = 60, random_seed: int = 0, backend: str = "torch"
) -> None:
    """Checks if config is set, initializes the Torch distributed cluster and sets random seeds.

    :param timeout: Time until initialization times out (default: ``60``).
    :param random_seed: Random seed for Torch, numpy, random (default: ``0``).
    :param backend: The backend that torch uses for distributed training. Either "torch"
        or "deepspeed" (default: ``torch``).

    :raises KeyError: Checks on environment variables failed.
    :raises RuntimeError: Checks on PyTorch distributed backend failed.
    """
    for env_variable in [
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "NCCL_BLOCKING_WAIT",  # NCCL_BLOCKING_WAIT ,NCCL_ASYNC_ERROR_HANDLING
    ]:
        if env_variable not in os.environ:
            raise KeyError(f"Environment variable {env_variable} not registered!")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch distributed needs a GPU cluster.")
    if not dist.is_available():
        raise RuntimeError("Torch distributed backend not accessible.")
    if not dist.is_nccl_available():
        raise RuntimeError("NCCL link not available on worker.")
    if backend == "deepspeed":
        deepspeed.init_process_group(timeout=datetime.timedelta(seconds=timeout))
    else:
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=timeout)
        )
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def _wrap_module_dispatcher(
    config: TorchDistributedConfig,
) -> Union[
    List[Type["MaggyDDPModuleWrapper"]],  # noqa: F821
    List[Type["MaggyFairScaleModuleWrapper"]],  # noqa: F821
    List[Type["MaggyDeepSpeedModuleWrapper"]],  # noqa: F821
    Type["MaggyDDPModuleWrapper"],  # noqa: F821
    Type["MaggyFairScaleModuleWrapper"],  # noqa: F821
    Type["MaggyDeepSpeedModuleWrapper"],  # noqa: F821
]:
    """Dispatcher for module wrapping.

    In case the user has passed a list of modules, converts each of the modules to be distributed.

    :param config: Experiment config.

    :returns: Either a list of wrapped modules or a single module.
    """
    if isinstance(config.module, list):
        wrapped_module = []
        for module in config.module:
            wrapped_module.append(
                _wrap_module(
                    module,
                    config.backend,
                    config.zero_lvl,
                    config.mixed_precision,
                    config.ds_config,
                )
            )
    else:
        wrapped_module = _wrap_module(
            config.module,
            config.backend,
            config.zero_lvl,
            config.mixed_precision,
            config.ds_config,
        )
    return wrapped_module


def _wrap_module(
    module: Type[torch.nn.Module],
    backend: str,
    zero_lvl: int,
    mixed_precision: bool,
    ds_config: dict,
) -> Union[
    Type["MaggyDDPModuleWrapper"],  # noqa: F821
    Type["MaggyFairScaleModuleWrapper"],  # noqa: F821
    Type["MaggyDeepSpeedModuleWrapper"],  # noqa: F821
]:
    """Wraps the module according to `backend`.

    :param module: Experiment config.
    :param backend: The backend engine used for training.
    :param zero_lvl: Sets the ZeRO optimization stages for `torch`.
    :param mixed_precision: Used to control the use of mixed precision training in `zero_lvl` 3.
    :param ds_config: DeepSpeed configuration dictionary if ds is enabled.

    :returns: Depending on the backend, returns a module that is a Maggy wrapper of either a PyTorch
        distributed module, a fairscale fully sharded module or a deepspeed engine.
    """
    if backend == "torch" and zero_lvl in [0, 1, 2]:
        module = get_maggy_ddp_wrapper(module)
    elif backend == "torch":
        module = get_maggy_fairscale_wrapper(module, mixed_precision)
    elif backend == "deepspeed":
        module = get_maggy_deepspeed_wrapper(module, ds_config)
    return module


def _sanitize_config(config: TorchDistributedConfig) -> None:
    if isinstance(config.module, list):
        for module in config.module:
            if not (isinstance(module, type) or callable(module)):
                raise TypeError(
                    """Passed module should be a class or a factory
                                callable."""
                )
    elif not (isinstance(config.module, type) or callable(config.module)):
        raise TypeError(
            """Passed module should be a class or a factory
                        callable."""
        )
    if config.backend == "torch":
        if config.ds_config:
            print(
                "Warning: DeepSpeed config passed for torch backend. LagomConfig will be discarded."
            )
        if config.zero_lvl not in [0, 1, 2, 3]:
            raise ValueError(
                f"DeepSpeed level has to be in [0,1,2,3], is {config.zero_lvl}."
            )
        return
    if config.backend == "deepspeed":
        _sanitize_ds_config(config)
        return
    raise ValueError(f"Unsupported backend {config.backend}.")


def _sanitize_ds_config(config: TorchDistributedConfig) -> None:
    if not config.ds_config:
        raise ValueError(
            """DeepSpeed ZeRO requires a configuration! For more information, see
            https://www.deepspeed.ai/docs/config-json/"""
        )
    if config.zero_lvl in [1, 2, 3]:
        print("Warning: Seperate ZeRO level set. Overwriting the config.")
        config.ds_config["zero_optimization"]["stage"] = config.zero_lvl
        config.zero_lvl = 0  # Avoid monkey patching after overwrite.
    elif config.zero_lvl != 0:
        raise ValueError("ZeRO level out of range! Zero accepts levels from 0 to 3")
    if config.ds_config["optimizer"]["type"] != "Adam":
        raise ValueError("ZeRO currently only supported with Adam optimizer.")
    # Currently Ninja JIT fails on Spark workers, so we force PyTorch optimizer in order to not
    # trigger the build of DeepSpeed ops. This should be resolved in future versions.
    config.ds_config["optimizer"]["params"]["torch_adam"] = True


def _monkey_patch_pytorch(zero_lvl: int) -> None:
    # Patch DataLoader to always be distributed.
    torch.utils.data.DataLoader = MaggyDataLoader
    if zero_lvl > 0:
        torch.optim.Adadelta = MaggyZeroAdadelta
        torch.optim.Adagrad = MaggyZeroAdagrad
        torch.optim.Adam = MaggyZeroAdam
        torch.optim.AdamW = MaggyZeroAdamW
        torch.optim.SparseAdam = MaggyZeroSparseAdam
        torch.optim.Adamax = MaggyZeroAdamax
        torch.optim.ASGD = MaggyZeroASGD
        torch.optim.LBFGS = MaggyZeroLBFGS
        torch.optim.RMSprop = MaggyZeroRMSprop
        torch.optim.Rprop = MaggyZeroRprop
        torch.optim.SGD = MaggyZeroSGD
