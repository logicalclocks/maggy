""" pytest fixtures that can be resued across tests. the filename needs to be conftest.py
"""

# make sure env variables are set correctly
import findspark  # this needs to be the first import
findspark.init()

import logging
import pytest

from pyspark import HiveContext
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.streaming import StreamingContext


def quiet_py4j():
    """ turn down spark logging for the test context """
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session")
def sc(request):
    """ fixture for creating a spark context
    Args:
        request: pytest.FixtureRequest object
    """
    conf = (SparkConf().setMaster("spark://Moritz-MacBook-Pro.local:7077")
                       .setAppName("pytest-pyspark-local-testing")
                       .set("spark.dynamicAllocation.maxExecutors", 2)
                       .set("spark.executor.instances", 2))
    scont = SparkContext(conf=conf)
    request.addfinalizer(lambda: scont.stop())

    quiet_py4j()
    return scont


@pytest.fixture(scope="session")
def hive_context(sc):
    """  fixture for creating a Hive Context. Creating a fixture enables it to be reused across all
        tests in a session
    Args:
        spark_context: spark_context fixture
    Returns:
        HiveContext for tests
    """
    return HiveContext(sc)


@pytest.fixture(scope="session")
def streaming_context(sc):
    return StreamingContext(sc, 1)