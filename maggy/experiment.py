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

"""
Experiment module used for running asynchronous optimization tasks.

The programming model is that you wrap the code containing the model
training inside a wrapper function.
Inside that wrapper function provide all imports and parts that make up your
experiment, see examples below. Whenever a function to run an experiment is
invoked it is also registered in the Experiments service along with the
provided information.
"""
import atexit
import time
from functools import singledispatch

from hops.experiment_impl.util import experiment_utils

from maggy import util
from maggy.core.lagom.lagom_optimization import lagom_optimization
from maggy.core.lagom.lagom_ablation import lagom_ablation
from maggy.core.lagom.lagom_distributed import lagom_distributed
from maggy.experiment_config import (
    OptimizationConfig,
    AblationConfig,
    DistributedConfig,
)


APP_ID = None
RUNNING = False
RUN_ID = 1
EXPERIMENT_JSON = {}


def lagom(train_fn, config):
    global APP_ID
    global RUNNING
    global RUN_ID
    job_start = time.time()
    try:
        if RUNNING:
            raise RuntimeError("An experiment is currently running.")
        RUNNING = True
        spark_context = util.find_spark().sparkContext
        APP_ID = str(spark_context.applicationId)
        result = lagom_wrapper(config, train_fn)  # Singledispatch uses first arg.
        return result
    except:  # noqa: E722
        _exception_handler(util.seconds_to_milliseconds(time.time() - job_start))
        raise
    finally:
        # cleanup spark jobs
        RUN_ID += 1
        RUNNING = False
        util.find_spark().sparkContext.setJobGroup("", "")


@singledispatch
def lagom_wrapper(config, train_fn):
    raise ValueError(
        "Invalid config type! Config is expected to be of type {}, {} or {}, \
                     but is of type {}".format(
            OptimizationConfig, AblationConfig, DistributedConfig, type(config)
        )
    )


@lagom_wrapper.register(OptimizationConfig)
def _(config, train_fn):
    return lagom_optimization(train_fn, config, APP_ID, RUN_ID)


@lagom_wrapper.register(AblationConfig)
def _(config, train_fn):
    return lagom_ablation(train_fn, config, APP_ID, RUN_ID)


@lagom_wrapper.register(DistributedConfig)
def _(config, train_fn):
    return lagom_distributed(train_fn, config, APP_ID, RUN_ID)


def _exception_handler(duration):
    """
    Handles exceptions during execution of an experiment

    :param duration: duration of the experiment until exception in milliseconds
    :type duration: int
    """
    try:
        global RUNNING
        global EXPERIMENT_JSON
        if RUNNING:
            EXPERIMENT_JSON["state"] = "FAILED"
            EXPERIMENT_JSON["duration"] = duration
            exp_ml_id = APP_ID + "_" + str(RUN_ID)
            experiment_utils._attach_experiment_xattr(
                exp_ml_id, EXPERIMENT_JSON, "FULL_UPDATE"
            )
    except Exception as err:
        util._log(err)


def _exit_handler():
    """
    Handles jobs killed by the user.
    """
    try:
        global RUNNING
        global EXPERIMENT_JSON
        if RUNNING:
            EXPERIMENT_JSON["status"] = "KILLED"
            exp_ml_id = APP_ID + "_" + str(RUN_ID)
            experiment_utils._attach_experiment_xattr(
                exp_ml_id, EXPERIMENT_JSON, "FULL_UPDATE"
            )
    except Exception as err:
        util._log(err)


atexit.register(_exit_handler)
