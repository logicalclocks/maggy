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
import calendar
import time
from functools import singledispatch
from typing import Callable

from maggy import util
from maggy.core.environment.singleton import EnvSing
from maggy.config import *
from maggy.core.experiment_driver import (
    HyperparameterOptDriver,
    AblationDriver,
    BaseDriver,
)


APP_ID = None
RUNNING = False
RUN_ID = 1
EXPERIMENT_JSON = {}


def lagom(train_fn: Callable, config) -> dict:
    """Launches a maggy experiment, which depending on 'config' can either
    be a hyperparameter optimization, an ablation study experiment or distributed
    training. Given a search space, objective and a model training procedure `train_fn`
    (black-box function), an experiment is the whole process of finding the
    best hyperparameter combination in the search space, optimizing the
    black-box function. Currently maggy supports random search and a median
    stopping rule.
    **lagom** is a Swedish word meaning "just the right amount".

    :param train_fn: User defined experiment containing the model training.
    :param config: An experiment configuration. For more information, see config.

    :returns: The experiment results as a dict.
    """
    global APP_ID
    global RUNNING
    global RUN_ID
    job_start = time.time()
    try:
        if RUNNING:
            raise RuntimeError("An experiment is currently running.")
        RUNNING = True
        APP_ID = str(calendar.timegm(time.gmtime()))
        APP_ID = "application_" + APP_ID + "_0001"
        APP_ID, RUN_ID = util.register_environment(APP_ID, RUN_ID)
        driver = lagom_driver(config, APP_ID, RUN_ID)
        return driver.run_experiment(train_fn, config)
    except:  # noqa: E722
        _exception_handler(util.seconds_to_milliseconds(time.time() - job_start))
        raise
    finally:
        # Clean up spark jobs
        RUN_ID += 1
        RUNNING = False


@singledispatch
def lagom_driver(config, app_id: int, run_id: int) -> None:
    """Dispatcher function for the experiment driver.

    Initializes the appropriate driver according to the config.

    :raises TypeError: Only gets called if no fitting config was found and
        raises an error.
    """
    raise TypeError(
        "Invalid config type! Config is expected to be of type {}, {}, {}, {} or {}, \
                     but is of type {}".format(
            HyperparameterOptConfig,
            AblationConfig,
            TorchDistributedConfig,
            TfDistributedConfig,
            BaseConfig,
            type(config),
        )
    )


@lagom_driver.register(HyperparameterOptConfig)
def _(
    config: HyperparameterOptConfig, app_id: int, run_id: int
) -> HyperparameterOptDriver:
    return HyperparameterOptDriver(config, app_id, run_id)


@lagom_driver.register(AblationConfig)
def _(config: AblationConfig, app_id: int, run_id: int) -> AblationDriver:
    return AblationDriver(config, app_id, run_id)


@lagom_driver.register(TorchDistributedConfig)
# Lazy import of DistributedDriver to avoid Torch import until necessary
def _(
    config: TorchDistributedConfig, app_id: int, run_id: int
) -> "TorchDistributedTrainingDriver":  # noqa: F821
    from maggy.core.experiment_driver.torch_distributed_training_driver import (
        TorchDistributedTrainingDriver,
    )

    return TorchDistributedTrainingDriver(config, app_id, run_id)


@lagom_driver.register(TfDistributedConfig)
# Lazy import of TfDistributedTrainingDriver to avoid Tensorflow import until necessary
def _(
    config: TfDistributedConfig, app_id: int, run_id: int
) -> "TfDistributedTrainingDriver":  # noqa: F821
    from maggy.core.experiment_driver.tf_distributed_training_driver import (
        TfDistributedTrainingDriver,
    )

    return TfDistributedTrainingDriver(config, app_id, run_id)


@lagom_driver.register(BaseConfig)
# Lazy import of BaseConfig
def _(config: BaseConfig, app_id: int, run_id: int) -> BaseDriver:
    from maggy.core.experiment_driver.base_driver import (
        BaseDriver,
    )

    return BaseDriver(config, app_id, run_id)


@lagom_driver.register(LagomConfig)
# Lazy import of LagomConfig
def _(config: LagomConfig, app_id: int, run_id: int) -> BaseDriver:
    from maggy.core.experiment_driver.base_driver import (
        BaseDriver,
    )

    return BaseDriver(config, app_id, run_id)


def _exception_handler(duration: int) -> None:
    """Handles exceptions during execution of an experiment.

    :param duration: Duration of the experiment until exception in milliseconds
    """
    try:
        global RUNNING
        global EXPERIMENT_JSON
        if RUNNING:
            EXPERIMENT_JSON["state"] = "FAILED"
            EXPERIMENT_JSON["duration"] = duration
            exp_ml_id = APP_ID + "_" + str(RUN_ID)
            EnvSing.get_instance().attach_experiment_xattr(
                exp_ml_id, EXPERIMENT_JSON, "FULL_UPDATE"
            )
    except Exception as err:
        util.log(err)


def _exit_handler() -> None:
    """Handles jobs killed by the user."""
    try:
        global RUNNING
        global EXPERIMENT_JSON
        if RUNNING:
            EXPERIMENT_JSON["status"] = "KILLED"
            exp_ml_id = APP_ID + "_" + str(RUN_ID)
            EnvSing.get_instance().attach_experiment_xattr(
                exp_ml_id, EXPERIMENT_JSON, "FULL_UPDATE"
            )
    except Exception as err:
        util.log(err)


atexit.register(_exit_handler)
