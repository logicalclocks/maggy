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

import torch
import torch.distributed as dist

import numpy as np

from maggy import util, tensorboard
from maggy.core import rpc
from maggy.core.reporter import Reporter
from maggy.distributed.patching import MaggyDataLoader
from maggy.core.environment.singleton import EnvSing

torch.utils.data.DataLoader = (
    MaggyDataLoader  # Patch data loader to always be distributed.
)


def prepare_function(
    app_id, run_id, train_fn, server_addr, hb_interval, secret, log_dir, **kwargs
):
    """
    Wraps the user supplied training function in order to be passed to the Spark Executors.

    Args:
        app_id (int): Maggy application ID.
        run_id (int): Maggy run ID.
        _ (object): Argument sink for experiment type to be compatible with other signatures.
        train_fn (Callable): Original training function.
        server_addr (str): IP of the Maggy worker registration RPC server.
        hb_interval (Union[float, int]): Worker heartbeat interval.
        secret (str): Secret string to authenticate messages.
        log_dir (str): Location of the logger file directory on the file system.
    """

    def wrapper_function(_):
        """
        Patched function from prepare_function factory.

        Args:
            _ (object): Necessary sink for the iterator given by Spark to the function upon foreach
                calls. Can safely be disregarded.
        """
        util.set_ml_id(app_id, run_id)
        partition_id, _ = util.get_partition_attempt_id()
        client = rpc.Client(server_addr, partition_id, 0, hb_interval, secret)
        log_file = log_dir + "/executor_" + str(partition_id) + ".log"

        reporter = Reporter(log_file, partition_id, 0, __builtin__.print)
        _setup_maggy_print(reporter)

        try:
            _register_with_servers(client, reporter, partition_id)
            tb_logdir, trial_log_file = _setup_logging(reporter, log_dir)

            client.await_reservations()
            config = client.get_torch_config()
            addr, port = config["host_port"].split(":")
            torch_config = {
                "MASTER_ADDR": addr,
                "MASTER_PORT": port,
                "WORLD_SIZE": str(config["num_executors"]),
                "RANK": str(partition_id),
                "NCCL_BLOCKING_WAIT": "1",
            }
            reporter.log(f"Torch config is {torch_config}")

            _setup_torch_env(torch_config)
            _init_cluster(timeout=60, random_seed=0)
            rank = int(torch_config["RANK"])
            model = torch.nn.parallel.DistributedDataParallel(kwargs["model"].cuda())

            reporter.log("Starting distributed training.", True)
            # device = torch.device(torch.cuda.current_device())
            sig = inspect.signature(train_fn)
            if sig.parameters.get("reporter", None):
                retval = train_fn(
                    model=model,
                    train_set=kwargs["train_set"],
                    test_set=kwargs["test_set"],
                    reporter=reporter,
                )
            else:
                retval = train_fn(
                    model=model,
                    train_set=kwargs["train_set"],
                    test_set=kwargs["test_set"],
                )
            if rank == 0:
                retval = util.handle_return_val(
                    retval, tb_logdir, "Metric", trial_log_file
                )

            reporter.log("Finished distributed training.", False)
            reporter.log("Final metric: {}".format(retval), False)
            client.finalize_metric(retval, reporter)
        except:  # noqa: E722
            reporter.log(traceback.format_exc(), False)
            raise
        finally:
            reporter.close_logger()
            client.stop()
            client.close()

    return wrapper_function


def _setup_maggy_print(reporter):
    """Modifies printing in the train function to also write to the logger.

    Args:
        reporter (Reporter): Reporter object responsible for logging.
    """
    builtin_print = __builtin__.print

    def maggy_print(*args, **kwargs):
        """Maggy custom print() function."""
        builtin_print(*args, **kwargs)
        reporter.log(" ".join(str(x) for x in args), True)

    __builtin__.print = maggy_print


def _register_with_servers(client, reporter, partition_id):
    """Registers own address with server and starts heartbeat protocol.

    Args:
        client (Client): Client for communication with the server.
        reporter (Reporter): Reporter responsible for heartbeat.
        partition_id (int): Executors partition ID from Sparks RDD.
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


def _get_open_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))  # Bind to 0 lets OS choose a free socket.
    port = sock.getsockname()[1]
    sock.close()
    return port


def _setup_logging(reporter, log_dir):
    """Sets up logging directories and files, registers with tensorboard.

    Args:
        reporter (Reporter): Reporter responsible for logging.
        log_dir (str): Log directory path on the file system.

    Returns:
        (tuple): Tuple containing:
            tb_logdir (str): Path of the tensorboard directory.
            trial_log_file (str): Path of the trial log file.
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
    tensorboard._register(tb_logdir)
    return tb_logdir, trial_log_file


def _setup_torch_env(torch_config):
    """Registers the Torch config as environment variables on the worker.

    Args:
        torch_config (dict): Dictionary containing the values of the variables.
    """
    for env_variable in [
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "NCCL_BLOCKING_WAIT",
    ]:
        os.environ[env_variable] = str(torch_config[env_variable])


def _init_cluster(timeout=60, random_seed=0):
    """Checks if config is set, initializes the Torch distributed cluster and sets random seeds.

    Args:
        timeout (:obj:'int', optional): Time until initialization times out. Defaults to 60.
        random_seed (:obj:'int', optional): Random seed for Torch, numpy, random. Defaults to 0.

    Raises:
        AssertionError: Checks on environment variables or Torch distributed backend failed.
    """
    for env_variable in [
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "NCCL_BLOCKING_WAIT",
    ]:
        assert (
            env_variable in os.environ
        ), f"Environment variable {env_variable} not registered."
    assert dist.is_available(), "Torch distributed backend not accessible."
    assert dist.is_nccl_available(), "NCCL link not available on worker."
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=timeout))
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
