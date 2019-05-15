import socket
import math
from pyspark.sql import SparkSession
from pyspark import TaskContext

from hops import util as hopsutil
from hops import hdfs as hopshdfs


def _get_experiments_dir(name):
    """Checks if a directory for the given maggy experiment `name` exists and
    if not creates it and returns the path.
    """
    base_dir = hopshdfs._get_experiments_dir()
    maggy_exp_dir = base_dir + '/maggy/' + str(name)

    if not hopshdfs.exists(maggy_exp_dir):
        hopshdfs.mkdir(maggy_exp_dir)

    return maggy_exp_dir

def _get_run_dir(name):
    """Checks the index of the latest run of an experiments and creates a new
    directory with the index incremented by one.
    """
    maggy_exp_dir = _get_experiments_dir(name)

    ls_exp_dir = hopshdfs.ls(maggy_exp_dir)

    if len(ls_exp_dir) == 0:
        run_dir = maggy_exp_dir + '/run.1'
        hopshdfs.mkdir(run_dir)
        return 1, run_dir
    elif len(ls_exp_dir) > 0:
        run_index = list(map(lambda x: int(x.rsplit('.', 1)[-1]), ls_exp_dir))
        max_run = max(run_index)
        new_run = max_run + 1
        run_dir = maggy_exp_dir + '/run.' + str(new_run)
        hopshdfs.mkdir(run_dir)
        return new_run, run_dir

def _get_runs(name):
    """Returns a list of hdfs paths for the runs of the experiment with `name`.
    """
    maggy_exp_dir = _get_experiments_dir(name)
    return hopshdfs.ls(maggy_exp_dir)

def _init_run(run_dir, app_id):
    app_dir = run_dir + '/' + app_id
    trials = app_dir + '/trials'
    logs = app_dir + '/logs'

    hopshdfs.mkdir(trials)
    hopshdfs.mkdir(logs)

    return app_dir, trials, logs

def num_executors():
    """
    Get the number of executors configured for Jupyter
    Returns:
        Number of configured executors for Jupyter
    """
    sc = hopsutil._find_spark().sparkContext
    try:
        return int(sc._conf.get('spark.dynamicAllocation.maxExecutors'))
    except:
        return int(sc._conf.get('spark.executor.instances'))

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

def print_trial_store(store):
    for _, value in store.items():
        print(value.to_json())

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