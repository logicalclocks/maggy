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

from maggy.experiment import experiment_python
from maggy.experiment import experiment_pyspark
from maggy.core import config as maggyconfig
from typing import Callable
from maggy.experiment_config import LagomConfig


def lagom(train_fn: Callable, config: LagomConfig) -> dict:

    """Entry point for Maggy experiment, this function passes the parameters to the lagom function
    depending whether the kernel is pyspark or python.
    **lagom** is a Swedish word meaning "just the right amount".

    :param train_fn: User defined experiment containing the model training.
    :param config: An experiment configuration. For more information, see experiment_config.

    :returns: The experiment results as a dict.
    """
    if maggyconfig.is_spark_available():
        return experiment_pyspark.lagom(train_fn, config)
    else:
        return experiment_python.lagom(train_fn, config)