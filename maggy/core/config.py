#
#   Copyright 2021 Logical Clocks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

import os
import tensorflow as tf

HOPSWORKS = "HOPSWORKS"
SPARK_ONLY = "SPARK_ONLY"

SPARK_AVAILABLE = None
MODE = None
TF_VERSION = None


def initialize():
    tf_full = tf.__version__.split(".")[0]
    # for building the docs since mock object doesn't mock int()
    global TF_VERSION
    global MODE
    if not isinstance(tf_full, str):
        TF_VERSION = 2
    else:
        TF_VERSION = int(tf_full)

    try:
        MODE = os.environ["HOPSWORKS_VERSION"]
        MODE = HOPSWORKS
        print("You are running maggy on Hopsworks.")
    except KeyError:
        MODE = SPARK_ONLY
        print("You are running maggy in pure Spark mode.")

    try:
        import pyspark

        # adding cause otherwise flake complains
        pyspark.TaskContext

        global SPARK_AVAILABLE
        SPARK_AVAILABLE = True
    except ModuleNotFoundError:
        SPARK_AVAILABLE = False
        print("Pyspark is not available, running maggy without spark.")
    finally:
        global SPARK_AVAILABLE
        SPARK_AVAILABLE = True


def is_spark_available():
    try:
        import pyspark

        # adding cause otherwise flake complains
        pyspark.TaskContext
    except ModuleNotFoundError:
        SPARK_AVAILABLE = False
        return False
    finally:
        global SPARK_AVAILABLE
        SPARK_AVAILABLE = True
    return True
