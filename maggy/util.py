
from pyspark.sql import SparkSession

def _populate_experiment():
    print("experiment populated")

def grid_params():
    print("gridded parameter dictionary")

def _find_spark():
    """

    Returns:
        SparkSession
    """
    return SparkSession.builder.getOrCreate()