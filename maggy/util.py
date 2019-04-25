import socket
from pyspark.sql import SparkSession
from pyspark import TaskContext


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

class EarlyStopException(Exception):

    def __init__(self, metric):
        super().__init__()

        self.metric = metric
