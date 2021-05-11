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
import random
import socket
from typing import Callable, Any, Tuple

import numpy as np
import tensorflow
import tensorflow as tf

from maggy import util
from maggy.core.tf_patching.tf_modules import get_wrapped_model
from maggy.experiment_config import TfDistributedConfig
from maggy.core.rpc import Client
from maggy.core.reporter import Reporter
from maggy.core.environment.singleton import EnvSing


def local_executor_fn(
    train_fn: Callable,
    config: TfDistributedConfig,
    app_id: int,
    run_id: int,
    log_dir: str,
) -> Callable:
    """Wraps the user supplied training function in order to be passed to the Spark Executors.

    :param train_fn: Original training function.
    :param config: Experiment config.
    :param app_id: Maggy application ID.
    :param run_id: Maggy run ID.
    :param log_dir: Location of the logger file directory on the file system.

    :returns: Patched function to execute on the Spark executors.
    """

    def wrapper_function(_: Any) -> None:
        """Patched function from tf_dist_executor_fn factory.

        :param _: Necessary catch for the iterator given by Spark to the
        function upon foreach calls. Can safely be disregarded.
        """

        EnvSing.get_instance().set_ml_id(app_id, run_id)

        log_file = log_dir + "/executor.log"

        with open(log_file, "w") as reporter:
            try:

                strategy = tf.distribute.MirroredStrategy

                print(f"Writing reports at {log_dir}")

                reporter.write(f"Distributed strategy is {type(strategy)} \n")

                model = _wrap_model(config, strategy)

                train_set, test_set = _consume_data(config)

                reporter.write("Starting training. \n")
                retval = train_fn(
                    model=model,
                    train_set=train_set,
                    test_set=test_set,
                    hparams=config.hparams,
                )

                print(
                    f"Final loss: {retval[0] if isinstance(retval, list) else retval}"
                )
                reporter.write(f"{str(retval)} \n")
            except:  # noqa: E722
                raise

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


def _init_seed(random_seed: int = 0) -> None:
    """Checks if config is set and sets random seeds.

    :param random_seed: Random seed for tensorflow, numpy, random (default: ``0``).

    :raises KeyError: Checks on environment variables failed.
    """
    if "TF_CONFIG" not in os.environ:
        raise KeyError("Environment variable TF_CONFIG not registered!")

    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


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


def _sanitize_init_model_params(model: tensorflow.keras.Model) -> None:
    assert isinstance(model, type) or callable(
        model
    ), """Passed model should be a
        class, not an instance."""


def _sanitize_init_strategy_params(
    strategy: tensorflow.distribute.MultiWorkerMirroredStrategy,
) -> None:
    assert isinstance(strategy, type) or callable(
        strategy
    ), """Passed strategy should be a
        class, not an instance."""


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

    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    data.with_options(options)

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
