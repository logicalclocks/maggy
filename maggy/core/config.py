#
#   Copyright 2020 Logical Clocks AB
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

mode = None
tf_full = tf.__version__.split(".")[0]
# for building the docs since mock object doesn't mock int()
if not isinstance(tf_full, str):
    tf_version = 2
else:
    tf_version = int(tf_full)

try:
    mode = os.environ["HOPSWORKS_VERSION"]
    mode = HOPSWORKS
    print("You are running maggy on Hopsworks.")
except KeyError:
    mode = SPARK_ONLY
    print("You are running maggy in pure Spark mode.")
