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

"""Utility helper module for maggy experiments.
"""
import math
import os
import json

import tensorflow as tf
import numpy as np

from maggy import constants, tensorboard
from maggy.core import exceptions
from maggy.core.environment.singleton import EnvSing
import maggy.core.config as mc

if mc.is_spark_available():
    from pyspark import TaskContext
    from pyspark.sql import SparkSession

DEBUG = True
APP_ID = None


def log(msg):
    """
    Generic log function (in case logging is changed from stdout later)

    :param msg: The msg to log
    :type msg: str
    """
    if DEBUG:
        print(msg)


def num_executors(sc):
    """
    Get the number of executors configured for Jupyter

    :param sc: The SparkContext to take the executors from.
    :type sc: [SparkContext
    :return: Number of configured executors for Jupyter
    :rtype: int
    """

    return EnvSing.get_instance().get_executors(sc)


def get_partition_attempt_id():
    """Returns partitionId and attemptNumber of the task context, when invoked
    on a spark executor.
    PartitionId is ID of the RDD partition that is computed by this task.
    The first task attempt will be assigned attemptNumber = 0, and subsequent
    attempts will have increasing attempt numbers.
    Returns:
        partitionId, attemptNumber -- [description]
    """
    if mc.is_spark_available():
        task_context = TaskContext.get()
        return task_context.partitionId(), task_context.attemptNumber()
    else:
        return 0, 0


def progress_bar(done, total):
    done_ratio = done / total
    progress = math.floor(done_ratio * 30)

    bar = "["

    for i in range(30):
        if i < progress:
            bar += "="
        elif i == progress:
            bar += ">"
        else:
            bar += "."

    bar += "]"
    return bar


def json_default_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(
            "Object of type {0}: {1} is not JSON serializable".format(type(obj), obj)
        )


def finalize_experiment(
    experiment_json,
    metric,
    app_id,
    run_id,
    state,
    duration,
    logdir,
    best_logdir,
    optimization_key,
):
    EnvSing.get_instance().finalize_experiment(
        experiment_json,
        metric,
        app_id,
        run_id,
        state,
        duration,
        logdir,
        best_logdir,
        optimization_key,
    )


def build_summary_json(logdir):
    """Builds the summary json to be read by the experiments service."""
    combinations = []
    env = EnvSing.get_instance()
    for trial in env.ls(logdir):
        if env.isdir(trial):
            return_file = trial + "/.outputs.json"
            hparams_file = trial + "/.hparams.json"
            if env.exists(return_file) and env.exists(hparams_file):
                metric_arr = env.convert_return_file_to_arr(return_file)
                hparams_dict = _load_hparams(hparams_file)
                combinations.append({"parameters": hparams_dict, "outputs": metric_arr})

    return json.dumps({"combinations": combinations}, default=json_default_numpy)


def _load_hparams(hparams_file):
    """Loads the HParams configuration from a hparams file of a trial."""

    hparams_file_contents = EnvSing.get_instance().load(hparams_file)
    hparams = json.loads(hparams_file_contents)

    return hparams


def handle_return_val(return_val, log_dir, optimization_key, log_file):
    """Handles the return value of the user defined training function."""
    env = EnvSing.get_instance()

    env.upload_file_output(return_val, log_dir)

    # Return type validation
    if not optimization_key:
        raise ValueError("Optimization key cannot be None.")
    if not return_val:
        raise exceptions.ReturnTypeError(optimization_key, return_val)
    if not isinstance(return_val, constants.USER_FCT.RETURN_TYPES):
        raise exceptions.ReturnTypeError(optimization_key, return_val)
    if isinstance(return_val, dict) and optimization_key not in return_val:
        raise KeyError(
            "Returned dictionary does not contain optimization key with the "
            "provided name: {}".format(optimization_key)
        )

    # validate that optimization metric is numeric
    if isinstance(return_val, dict):
        opt_val = return_val[optimization_key]
    else:
        opt_val = return_val
        return_val = {optimization_key: opt_val}

    if not isinstance(opt_val, constants.USER_FCT.NUMERIC_TYPES):
        raise exceptions.MetricTypeError(optimization_key, opt_val)

    # for key, value in return_val.items():
    #    return_val[key] = value if isinstance(value, str) else str(value)

    return_val["log"] = log_file.replace(env.project_path(), "")

    return_file = log_dir + "/.outputs.json"
    env.dump(json.dumps(return_val, default=json_default_numpy), return_file)

    metric_file = log_dir + "/.metric"
    env.dump(json.dumps(opt_val, default=json_default_numpy), metric_file)

    return opt_val


def clean_dir(clean_dir, keep=[]):
    """Deletes all files in a directory but keeps a few."""
    env = EnvSing.get_instance()

    if not env.isdir(clean_dir):
        raise ValueError(
            "{} is not a directory. Use `hops.hdfs.delete()` to delete single "
            "files.".format(clean_dir)
        )
    for path in env.ls(clean_dir):
        if path not in keep:
            env.delete(path, recursive=True)


def validate_ml_id(app_id, run_id):
    """Validates if there was an experiment run previously from the same app id
    but from a different experiment (e.g. hops-util-py vs. maggy) module.
    """
    try:
        prev_ml_id = os.environ["ML_ID"]
    except KeyError:
        return app_id, run_id
    prev_app_id, _, prev_run_id = prev_ml_id.rpartition("_")
    if prev_run_id == prev_ml_id:
        # means there was no underscore found in string
        raise ValueError(
            "Found a previous ML_ID with wrong format: {}".format(prev_ml_id)
        )
    if prev_app_id == app_id and int(prev_run_id) >= run_id:
        return app_id, (int(prev_run_id) + 1)
    return app_id, run_id


def set_ml_id(app_id, run_id):
    """Sets the environment variables 'HOME' and 'ML_ID' to register the experiment.

    Args:
        app_id (int): Maggy App ID.
        run_id (int): Maggy experiment run ID.
    """
    os.environ["HOME"] = os.getcwd()
    os.environ["ML_ID"] = str(app_id) + "_" + str(run_id)


def find_spark():
    """
    Returns: SparkSession
    """
    if mc.is_spark_available():
        return SparkSession.builder.getOrCreate()
    else:
        return None


def seconds_to_milliseconds(time):
    """
    Returns: time converted from seconds to milliseconds
    """
    return int(round(time * 1000))


def time_diff(t0, t1):
    """
    Args:
        :t0: start time in seconds
        :t1: end time in seconds
    Returns: string with time difference (i.e. t1-t0)
    """
    minutes, seconds = divmod(t1 - t0, 60)
    hours, minutes = divmod(minutes, 60)
    return "%d hours, %d minutes, %d seconds" % (hours, minutes, seconds)


def register_environment(app_id, run_id):
    """Validates IDs and creates an experiment folder in the fs.

    Args:
        :app_id: Application ID
        :run_id: Current experiment run ID

    Returns: (app_id, run_id) with the updated IDs.
    """

    app_id, run_id = validate_ml_id(app_id, run_id)
    set_ml_id(app_id, run_id)
    # Create experiment directory.
    EnvSing.get_instance().create_experiment_dir(app_id, run_id)
    tensorboard._register(EnvSing.get_instance().get_logdir(app_id, run_id))
    return app_id, run_id


def populate_experiment(config, app_id, run_id, exp_function):
    """Creates a dictionary with the experiment information.

    Args:
        :config: Experiment config object
        :app_id: Application ID
        :run_id: Current experiment run ID
        :exp_function: Name of experiment driver.

    Returns:
        :experiment_json: Dictionary with config info on the experiment.
    """
    try:
        direction = config.direction
    except AttributeError:
        direction = "N/A"
    try:
        opt_key = config.optimization_key
    except AttributeError:
        opt_key = "N/A"
    experiment_json = EnvSing.get_instance().populate_experiment(
        config.name,
        exp_function,
        "MAGGY",
        None,
        config.description,
        app_id,
        direction,
        opt_key,
    )
    exp_ml_id = str(app_id) + "_" + str(run_id)
    experiment_json = EnvSing.get_instance().attach_experiment_xattr(
        exp_ml_id, experiment_json, "INIT"
    )
    return experiment_json


def num_physical_devices():
    """Returns the number of physical devices using Tensorflow.config.list_physical_devices() function.

    Returns:
        :int: number of physical devices.
    """
    return len(tf.config.list_physical_devices())


def set_app_id(app_id):
    """Sets the app_id if it's non None, this function is used when the kernel is python.

    Args:
        :app_id: the app id for the experiment
    """
    global APP_ID
    if APP_ID is None:
        APP_ID = app_id


def get_metric_value(return_dict, metric_key):
    if return_dict and metric_key:
        assert (
            metric_key in return_dict.keys()
        ), "Supplied metric_key {} is not in returned dict {}".format(
            metric_key, return_dict
        )
        return return_dict[metric_key]
    elif (
        return_dict is not None
        and len(return_dict.keys()) == 2
        and "metric" in return_dict.keys()
    ):
        return return_dict["metric"]
    else:
        return None
