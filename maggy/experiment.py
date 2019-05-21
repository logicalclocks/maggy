"""
Experiment module used for running asynchronous optimization tasks.

The programming model is that you wrap the code containing the model
training inside a wrapper function.
Inside that wrapper function provide all imports and parts that make up your
experiment, see examples below. Whenever a function to run an experiment is
invoked it is also registered in the Experiments service along with the
provided information.
"""
import socket
import os
import json
import atexit
from datetime import datetime

from maggy import util, tensorboard
from maggy.core import rpc, trialexecutor, ExperimentDriver
from maggy.trial import Trial

from hops import util as hopsutil
from hops import hdfs as hopshdfs

app_id = None
running = False
run_id = None
elastic_id = 1
experiment_json = None


def lagom(map_fun, searchspace, optimizer, direction, num_trials, name, hb_interval=1, es_policy='median', es_interval=300, es_min=10, description=''):
    """Launches a maggy experiment for hyperparameter optimization.

    Given a search space, objective and a model training procedure `map_fun`
    (black-box function), an experiment is the whole process of finding the
    best hyperparameter combination in the search space, optimizing the
    black-box function. Currently maggy supports random search and a median
    stopping rule.

    **lagom** is a Swedish word meaning "just the right amount".

    :param map_fun: User defined experiment containing the model training.
    :type map_fun: function
    :param searchspace: A maggy Searchspace object from which samples are drawn.
    :type searchspace: Searchspace
    :param optimizer: The optimizer is the part generating new trials.
    :type optimizer: str, AbstractOptimizer
    :param direction: If set to ‘max’ the highest value returned will
        correspond to the best solution, if set to ‘min’ the opposite is true.
    :type direction: str
    :param num_trials: the number of trials to evaluate given the search space,
        each containing a different hyperparameter combination
    :type num_trials: int
    :param name: A user defined experiment identifier.
    :type name: str
    :param hb_interval: The heartbeat interval in secondss from trial executor
        to experiment driver, defaults to 1
    :type hb_interval: int, optional
    :param es_policy: The earlystopping policy, defaults to 'median'
    :type es_policy: str, optional
    :param es_interval: Frequency interval in seconds to check currently
        running trials for early stopping, defaults to 300
    :type es_interval: int, optional
    :param es_min: Minimum number of trials finalized before checking for
        early stopping, defaults to 10
    :type es_min: int, optional
    :param description: A longer description of the experiment.
    :type description: str, optional
    :raises RuntimeError: An experiment is currently running.
    :return: A dictionary indicating the best trial and best hyperparameter
        combination with it's performance metric
    :rtype: dict
    """
    assert num_trials > 0, "number of trials should be greater than zero"

    global running

    if running:
        raise RuntimeError("An experiment is currently running.")

    try:
        global app_id
        global experiment_json
        global elastic_id
        global run_id
        running = True

        sc = hopsutil._find_spark().sparkContext
        app_id = str(sc.applicationId)
        app_dir = ''

        # Create the root dir if not existing
        _ = util._get_experiments_dir(name)
        # get run_id and run_dir
        run_id, run_dir = util._get_run_dir(name)
        # set elastic id to run_id
        elastic_id = run_id
        # trial dir will be for tensorboard
        app_dir, trial_dir, log_dir = util._init_run(run_dir, app_id)
        tensorboard._register(trial_dir)
        #tensorboard.write_hparams_proto(trial_dir, searchspace)
        #hopshdfs.dump('writing proto buf worked', log_dir+'/maggy.log')

        num_executors = util.num_executors()

        if num_executors > num_trials:
            num_executors = num_trials

        nodeRDD = sc.parallelize(range(num_executors), num_executors)

        # start experiment driver
        exp_driver = ExperimentDriver(searchspace, optimizer, direction,
            num_trials, name, num_executors, hb_interval, es_policy,
            es_interval, es_min, description, app_dir, log_dir, trial_dir)

        # Make SparkUI intuitive by grouping jobs
        sc.setJobGroup("Maggy Experiment", "{}".format(name))
        exp_driver._log("Started Maggy Experiment: {0}, run {1}".format(name, run_id))

        exp_driver.init()

        server_addr = exp_driver.server_addr

        experiment_json = exp_driver.json(sc)
        hopsutil._put_elastic(hopshdfs.project_name(), app_id, run_id,
            experiment_json)

        # Force execution on executor, since GPU is located on executor
        job_start = datetime.now()
        nodeRDD.foreachPartition(trialexecutor._prepare_func(app_id, run_id,
            map_fun, server_addr, hb_interval, exp_driver._secret, app_dir))
        job_end = datetime.now()

        result = exp_driver.finalize(job_start, job_end)

        experiment_json = exp_driver.json(sc)
        hopsutil._put_elastic(hopshdfs.project_name(), app_id, elastic_id,
            experiment_json)

        exp_driver._log("Finished Experiment")

    except:
        _exception_handler()
        raise
    finally:
        # cleanup spark jobs
        exp_driver.stop()
        elastic_id +=1
        running = False
        sc.setJobGroup("", "")

    return result

def _exception_handler():
    """

    Returns:

    """
    global running
    global experiment_json
    if running and experiment_json != None:
        experiment_json = json.loads(experiment_json)
        experiment_json['status'] = "FAILED"
        experiment_json['finished'] = datetime.now().isoformat()
        experiment_json = json.dumps(experiment_json)
        hopsutil._put_elastic(hopshdfs.project_name(), app_id, elastic_id, experiment_json)

def _exit_handler():
    """

    Returns:

    """
    global running
    global experiment_json
    if running and experiment_json != None:
        experiment_json = json.loads(experiment_json)
        experiment_json['status'] = "KILLED"
        experiment_json['finished'] = datetime.now().isoformat()
        experiment_json = json.dumps(experiment_json)
        hopsutil._put_elastic(hopshdfs.project_name(), app_id, elastic_id, experiment_json)

atexit.register(_exit_handler)
