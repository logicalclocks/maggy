
import os

HOPSWORKS = "HOPSWORKS"
SPARK_ONLY = "SPARK_ONLY"

mode = None

try:
    mode = os.environ['HOPSWORKS_VERSION']
    mode = HOPSWORKS
    print("You are running maggy on Hopsworks.")
except KeyError:
    mode = SPARK_ONLY
    print("You are running maggy in pure Spark mode.")