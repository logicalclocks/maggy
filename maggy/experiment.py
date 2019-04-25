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
import datetime

from maggy import util
from maggy.core import rpc
from maggy.core import trialexecutor
from maggy.core import ExperimentDriver
from maggy.trial import Trial

app_id = None
running = False

# TODO: this should be inferred from HDFS checkpoints
run_id = 0

elastic_id = 1
experiment_json = None
driver_tensorboard_hdfs_path = None


def launch(map_fun, searchspace, optimizer, direction, num_trials, name, hb_interval=1, es_policy='median', es_interval=300, es_min=10):
    """Launches a maggy experiment for hyperparameter optimization.

    Given a search space, objective and a model training procedure `map_fun`
    (black-box function), an experiment is the whole process of finding the
    best hyperparameter combination in the search space, optimizing the
    black-box function. Currently maggy supports random search and a median
    stopping rule.

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
    :raises RuntimeError: An experiment is currently running.
    :return: A dictionary indicating the best trial and best hyperparameter
        combination with it's performance metric
    :rtype: dict
    """
    assert num_trials > 0, "number of trials should be greater than zero"

    global running
    # move run_id to the class of the optimizer
    # global run_id
    if running:
        raise RuntimeError("An experiment is currently running.")

    try:
        global app_id
        global experiment_json
        global elastic_id
        running = True

        sc = util._find_spark().sparkContext
        app_id = str(sc.applicationId)

        num_executors = util.num_executors()

        if num_executors > num_trials:
            num_executors = num_trials

        nodeRDD = sc.parallelize(range(num_executors), num_executors)

        # Make SparkUI intuitive by grouping jobs
        sc.setJobGroup("Maggy Experiment", "{}".format(name))
        print("Started Maggy Experiment: ", "{}".format(name))

        # start experiment driver
        exp_driver = ExperimentDriver(searchspace, optimizer, direction,
            num_trials, name, num_executors, hb_interval, es_policy,
            es_interval, es_min)

        exp_driver.init()

        server_addr = exp_driver.server_addr

        # Force execution on executor, since GPU is located on executor
        job_start = datetime.datetime.now()
        nodeRDD.foreachPartition(trialexecutor._prepare_func(app_id, run_id,
            map_fun, server_addr, hb_interval))
        job_end = datetime.datetime.now()

        result = exp_driver.finalize(job_start, job_end)

        print("Finished Experiment \n")

    except:
        raise
    finally:
        # cleanup spark jobs
        exp_driver.stop()
        running = False
        sc.setJobGroup("", "")

    return result
