
import os
import tensorflow as tf

HOPSWORKS = "HOPSWORKS"
SPARK_ONLY = "SPARK_ONLY"

mode = None
tf_version = int(tf.__version__.split(".")[0])

try:
    mode = os.environ['HOPSWORKS_VERSION']
    mode = HOPSWORKS
    print("You are running maggy on Hopsworks.")
except KeyError:
    mode = SPARK_ONLY
    print("You are running maggy in pure Spark mode.")