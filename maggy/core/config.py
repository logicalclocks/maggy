
import os

mode = None

try:
    mode = os.environ['HOPSWORKS_VERSION']
    print("You are running maggy on Hopsworks.")
    import hops.util as hopsutil
    import hops.hdfs as hopshdfs
except KeyError:
    print("You are running maggy in pure Spark mode.")