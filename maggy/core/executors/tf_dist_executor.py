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
import json
import traceback
import os
import socket
from typing import Callable, Any, Tuple

import tensorflow as tf

from maggy import util
from maggy.core.tf_patching.tf_modules import get_wrapped_model
from maggy.experiment_config import TfDistributedConfig
from maggy.core.rpc import Client
from maggy.core.reporter import Reporter
from maggy.core.environment.singleton import EnvSing


def dist_executor_fn(
    train_fn: Callable,
    config: TfDistributedConfig,
    num_executors: int,
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
        """Patched function from tf_dist_executor_fn factory.

        :param _: Necessary catch for the iterator given by Spark to the
        function upon foreach calls. Can safely be disregarded.
        """

        EnvSing.get_instance().set_ml_id(app_id, run_id)
        partition_id, _ = util.get_partition_attempt_id()
        client = Client(server_addr, partition_id, 0, hb_interval, secret)
        log_file = log_dir + "/executor_" + str(partition_id) + ".log"

        reporter = Reporter(log_file, partition_id, 0, __builtin__.print)
        builtin_print = __builtin__.print

        def maggy_print(*args, **kwargs):
            builtin_print(*args, **kwargs)
            reporter.log(" ".join(str(x) for x in args), True)

        __builtin__.print = maggy_print

        try:
            _register_with_servers(client, reporter, partition_id)
            tb_logdir, trial_log_file = _setup_logging(reporter, log_dir)
            reporter.log("Awaiting worker reservations.")
            client.await_reservations()
            reporter.log("Reservations complete, configuring Tensorflow.")
            reservations = client.get_message("RESERVATIONS")
            if not reservations:
                reporter.log("Tensorflow registration failed, exiting from all tasks.")
                return

            workers_host_port = []
            for i in list(reservations):
                workers_host_port.append(reservations[i]["host_port"])

            tf_config = {
                "cluster": {"worker": workers_host_port},
                "task": {"index": int(partition_id), "type": "worker"},
            }
            reporter.log(f"Tensorflow config is {tf_config}")

            strategy = tf.distribute.MultiWorkerMirroredStrategy
            model = _wrap_model(config, strategy)

            train_set, test_set = _consume_data(config)

            config.train_set = _shard_data(
                train_set,
                config.hparams["train_batch_size"],
                len(reservations),
                partition_id,
            )

            config.test_set = _shard_data(
                test_set,
                config.hparams["test_batch_size"],
                len(reservations),
                partition_id,
            )

            reporter.log(f"index of slice {partition_id}")
            reporter.log("Starting distributed training.")
            sig = inspect.signature(train_fn)

            if sig.parameters.get("reporter", None):
                retval = train_fn(
                    model=model,
                    train_set=config.train_set,
                    test_set=config.test_set,
                    reporter=reporter,
                    hparams=config.hparams,
                )
            else:
                retval = train_fn(
                    model=model,
                    train_set=config.train_set,
                    test_set=config.test_set,
                    hparams=config.hparams,
                )

            # Set retval to work with util.handle_return_value,
            # if there is more than 1 metrics, retval will be a list and
            # retval[0] will contain the final loss
            retval = {"Metric": retval[0] if isinstance(retval, list) else retval}
            retval = util.handle_return_val(retval, tb_logdir, "Metric", trial_log_file)
            reporter.log("Finished distributed training.")
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


def _setup_tf_config(tf_config: dict) -> None:
    """Registers the TF_CONFIG environment variables on the worker.

    :param tf_config: Dictionary containing the values of the variables.
    """
    os.environ["TF_CONFIG"] = json.dumps(tf_config)


def _wrap_model(config, strategy):
    """Wraps the model according to `backend`.

    :param config: Experiment config.

    :returns: Returns a tensorflow model wrapped class of config.model.
    """
    # Instantiate model on executor in case its too large for pickle and sent as a class.
    _sanitize_init_model_params(config.model)
    _sanitize_init_strategy_params(strategy)
    model = get_wrapped_model(config.model, strategy())

    return model


def _sanitize_init_model_params(model: tf.keras.Model) -> None:
    assert isinstance(model, type) or callable(
        model
    ), "Passed model should be a class, not an instance."


def _sanitize_init_strategy_params(
    strategy: tf.distribute.MultiWorkerMirroredStrategy,
) -> None:
    assert isinstance(strategy, type) or callable(
        strategy
    ), "Passed strategy should be a class, not an instance."


def _shard_data(data, batch_size, num_shards, index):
    """Returns the index slice of the train_set, given the number of shards.
    If the data is not a tensor, it will be converted to tensor.

    :param data: Dataset to shard.
    :param num_shards: Number of slices to shard the data on.
    :param index:

    :returns: The slice of data at index.
    """

    # Wrap data in Dataset objects.
    data = tf.data.Dataset.from_tensor_slices(data)

    # The batch size must now be set on the Dataset objects.
    data = data.batch(batch_size)

    if index >= num_shards:
        raise RuntimeError(
            f"index ({index}) must be smaller than num_shard ({num_shards})"
        )

    data = data.shard(num_shards, int(index))

    return data


def _consume_data(config):
    """Load and return the training and test datasets from config file. If the config.train_set and config.test_set are
    strings they are assumed as path, the functions check if the files or directories exists, if they exists then it
    will run the function in config.process_data, with paramneters config.train_set and config.test_set and return the
    result.
    If the config.train_set and cofig.test_set are not strings but anything else (like a List, nparray, tf.data.Dataset)
    they will returned as they are.
    The types of config.train_set and config.test_set have to be the same.


    :param config: the experiment configuration dictionary

    :returns: train_set and test_set

    :raises TypeError: if the config.train_set and config.test_set are of different type
    :raises TypeError: if the process_data function is missing or cannot process the data
    :raises FileNotFoundError: in case config.train_set or config.test_set are not found
    """

    train_set = config.train_set
    test_set = config.test_set
    process_data = config.process_data

    if type(train_set) != type(test_set):
        raise TypeError(
            f"The train_set and test_set types are different but must be the same, "
            f"provided {type(train_set)} and {type(train_set)}"
        )

    data_type = type(train_set)
    if data_type == str:
        env = EnvSing.get_instance()

        if (env.isdir(train_set) or env.exists(train_set)) and (
            env.isdir(test_set) or env.exists(test_set)
        ):
            try:
                return process_data(train_set, test_set)
            except TypeError:
                raise TypeError(
                    (
                        f"process_data function missing in config, "
                        f"please provide a function that takes 2 arguments, "
                        f"train_set and test_set, read the datasets and "
                        f"return 2 tuples (X_train, y_train), (X_test, y_test). "
                        f"config: {config}"
                    )
                )
        else:
            if not env.isdir(train_set):
                assert env.exists(train_set), FileNotFoundError(
                    f"{train_set} does not exists."
                )
            if not env.isdir(test_set):
                assert env.exists(test_set), FileNotFoundError(
                    f"{test_set} does not exists."
                )
            raise RuntimeError(
                f"config.train_set: {config.train_set} and/or "
                f"config.test_set: {config.test_set} do not exists"
            )
    else:  # type is tf.data.Dataset
        return train_set, test_set
