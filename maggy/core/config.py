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

import tensorflow as tf

SPARK_AVAILABLE = None
try:
    from pyspark.sql import SparkSession  # noqa: F401

    SPARK_AVAILABLE = True
except ModuleNotFoundError:
    SPARK_AVAILABLE = False

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

    print("Detected Kernel: Python.") if not SPARK_AVAILABLE else print(
        "Detected Kernel: Spark."
    )


def is_spark_available():
    return SPARK_AVAILABLE
