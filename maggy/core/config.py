
import os
import tensorflow as tf

HOPSWORKS = "HOPSWORKS"
SPARK_ONLY = "SPARK_ONLY"

mode = None
tf_full = tf.__version__.split(".")[0]
# for building the docs since mock object doesn't mock int()
if not isinstance(tf_full, str):
    tf_version = 2
else:
    tf_version = int(tf_full)

try:
    mode = os.environ['HOPSWORKS_VERSION']
    mode = HOPSWORKS
    print("You are running maggy on Hopsworks.")
except KeyError:
    mode = SPARK_ONLY
    print("You are running maggy in pure Spark mode.")