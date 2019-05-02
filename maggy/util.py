import socket
import hops.util as hopsutil
from pyspark.sql import SparkSession
from pyspark import TaskContext


def _get_directories(name):
    """Checks if experiment directories exist in HDFS and if not creates them
    """
    pass

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
