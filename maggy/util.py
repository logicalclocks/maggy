import socket
from pyspark.sql import SparkSession
from pyspark import TaskContext

from maggy.core import config

if config.mode is config.HOPSWORKS:
    import hops.hdfs as hopshdfs


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
    results = app_dir + '/result'

    hopshdfs.mkdir(trials)
    hopshdfs.mkdir(logs)
    hopshdfs.mkdir(results)

    return app_dir, trials, logs, results

def _find_spark():
    """
    Returns:
        SparkSession
    """
    return SparkSession.builder.getOrCreate()

def _get_ip_address():
    """
    Simple utility to get host IP address
    Returns:
        x
    """
    try:
	    _, _, _, _, addr = socket.getaddrinfo(socket.gethostname(),
                                              None,
                                              socket.AF_INET,
                                              socket.SOCK_STREAM)[0]
	    return addr[0]
    except:
	    return socket.gethostbyname(socket.getfqdn())

def num_executors():
    """
    Get the number of executors configured for Jupyter
    Returns:
        Number of configured executors for Jupyter
    """
    sc = _find_spark().sparkContext
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

def _time_diff(task_start, task_end):
    """
    Args:
        :task_start:
        :tast_end:
    Returns:
    """
    time_diff = task_end - task_start

    seconds = time_diff.seconds

    if seconds < 60:
        return str(int(seconds)) + ' seconds'
    elif seconds == 60 or seconds <= 3600:
        minutes = float(seconds) / 60.0
        return str(int(minutes)) + ' minutes, ' + str((int(seconds) % 60)) + ' seconds'
    elif seconds > 3600:
        hours = float(seconds) / 3600.0
        minutes = (hours % 1) * 60
        return str(int(hours)) + ' hours, ' + str(int(minutes)) + ' minutes'
    else:
        return 'unknown time'