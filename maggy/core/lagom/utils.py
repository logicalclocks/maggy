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

from functools import singledispatch

from hops.experiment_impl.util import experiment_utils

from maggy import util, tensorboard
from maggy.experiment_config import (
    OptimizationConfig,
    AblationConfig,
    DistributedConfig,
)


def register_environment(app_id, run_id, spark_context):
    app_id = str(spark_context.applicationId)
    app_id, run_id = util._validate_ml_id(app_id, run_id)
    util.set_ml_id(app_id, run_id)
    # Create experiment directory.
    experiment_utils._create_experiment_dir(app_id, run_id)
    tensorboard._register(experiment_utils._get_logdir(app_id, run_id))
    return app_id, run_id


@singledispatch
def populate_experiment(config, app_id, run_id, exp_function=None):
    raise ValueError("Invalid config type!")


@populate_experiment.register(OptimizationConfig)
@populate_experiment.register(AblationConfig)
def _(config, app_id, run_id, exp_function):
    experiment_json = experiment_utils._populate_experiment(
        config.name,
        exp_function,
        "MAGGY",
        None,
        config.description,
        app_id,
        config.direction,
        config.optimization_key,
    )
    exp_ml_id = app_id + "_" + str(run_id)
    experiment_json = experiment_utils._attach_experiment_xattr(
        exp_ml_id, experiment_json, "INIT"
    )
    return experiment_json


@populate_experiment.register(DistributedConfig)
def _(config, app_id, run_id):
    experiment_json = experiment_utils._populate_experiment(
        config.name,
        "torch_dist",
        "MAGGY",
        None,
        config.description,
        app_id,
        "N/A",
        "N/A",
    )
    exp_ml_id = app_id + "_" + str(run_id)
    experiment_json = experiment_utils._attach_experiment_xattr(
        exp_ml_id, experiment_json, "INIT"
    )
    return experiment_json
