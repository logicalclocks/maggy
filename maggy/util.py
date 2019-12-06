import math
import json
import numpy as np
from pyspark import TaskContext

from hops import util as hopsutil
from hops import hdfs as hopshdfs
from hops.experiment_impl.util import experiment_utils

from maggy import constants
from maggy.core import exceptions

DEBUG = True


def _log(msg):
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
    sc = hopsutil._find_spark().sparkContext
    try:
        return int(sc._conf.get("spark.dynamicAllocation.maxExecutors"))
    except:
        raise RuntimeError(
            'Failed to find spark.dynamicAllocation.maxExecutors property, \
            please select your mode as either Experiment, Parallel \
            Experiments or Distributed Training.')

def get_partition_attempt_id():
    """Returns partitionId and attemptNumber of the task context, when invoked
    on a spark executor.
    PartitionId is ID of the RDD partition that is computed by this task.
    The first task attempt will be assigned attemptNumber = 0, and subsequent
    attempts will have increasing attempt numbers.
    Returns:
        partitionId, attemptNumber -- [description]
    """
    task_context = TaskContext.get()
    return task_context.partitionId(), task_context.attemptNumber()

def _progress_bar(done, total):

    done_ratio = done/total
    progress = math.floor(done_ratio * 30)

    bar = '['

    for i in range(30):
        if i < progress:
            bar += '='
        elif i == progress:
            bar += '>'
        else:
            bar += '.'

    bar += ']'
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
            "Object of type {0}: {1} is not JSON serializable"
            .format(type(obj), obj))

def _finalize_experiment(
        experiment_json, metric, app_id, run_id, state, duration, logdir,
        best_logdir, optimization_key):
    """[summary]

    :param experiment_json: [description]
    :type experiment_json: [type]
    :param metric: [description]
    :type metric: [type]
    :param app_id: [description]
    :type app_id: [type]
    :param run_id: [description]
    :type run_id: [type]
    :param state: [description]
    :type state: [type]
    :param duration: [description]
    :type duration: [type]
    :param logdir: [description]
    :type logdir: [type]
    :param best_logdir: [description]
    :type best_logdir: [type]
    :param optimization_key: [description]
    :type optimization_key: [type]
    :return: [description]
    :rtype: [type]
    """

    outputs = _build_summary_json(logdir)

    if outputs:
        hopshdfs.dump(outputs, logdir + '/.summary.json')

    if best_logdir:
        experiment_json['bestDir'] = best_logdir[len(hopshdfs.project_path()):]
    experiment_json['optimizationKey'] = optimization_key
    experiment_json['metric'] = metric
    experiment_json['state'] = state
    experiment_json['duration'] = duration

    experiment_utils._attach_experiment_xattr(
        app_id, run_id, experiment_json, 'REPLACE')

def _build_summary_json(logdir):

    combinations = []

    for trial in hopshdfs.ls(logdir):
        if hopshdfs.isdir(trial):
            return_file = trial + '/.outputs.json'
            hparams_file = trial + '/.hparams.json'
            if hopshdfs.exists(return_file) and hopshdfs.exists(hparams_file):
                metric_arr = experiment_utils._convert_return_file_to_arr(
                    return_file)
                hparams_dict = _load_hparams(hparams_file)
                combinations.append(
                    {'parameters': hparams_dict, 'metrics': metric_arr})

    return json.dumps(
        {'combinations': combinations}, default=json_default_numpy)

def _load_hparams(hparams_file):
    """[summary]

    :param hparams_file: [description]
    :type hparams_file: [type]
    :return: [description]
    :rtype: [type]
    """
    hparams_file_contents = hopshdfs.load(hparams_file)
    hparams = json.loads(hparams_file_contents)

    return hparams

def _handle_return_val(return_val, log_dir, optimization_key, log_file):
    experiment_utils._upload_file_output(return_val, log_dir)

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
            "provided name: {}".format(optimization_key))

    # validate that optimization metric is numeric
    if isinstance(return_val, dict):
        opt_val = return_val[optimization_key]
    else:
        opt_val = return_val
        return_val = {optimization_key: opt_val}

    if not isinstance(opt_val, constants.USER_FCT.NUMERIC_TYPES):
        raise exceptions.MetricTypeError(optimization_key, opt_val)

    #for key, value in return_val.items():
    #    return_val[key] = value if isinstance(value, str) else str(value)

    return_val['log'] = log_file

    return_file = log_dir + '/.outputs.json'
    hopshdfs.dump(json.dumps(return_val, default=json_default_numpy), return_file)

    metric_file = log_dir + '/.metric'
    hopshdfs.dump(json.dumps(opt_val, default=json_default_numpy), metric_file)

    return opt_val
