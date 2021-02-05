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
import time

from hops import util as hopsutil
from hops.experiment_impl.util import experiment_utils

from maggy import util, tensorboard
from maggy.core.experiment_driver.OptimizationDriver import OptimizationDriver
from maggy.core.executors.trial_executor import prepare_function


def lagom_optimization(train_fn, config, APP_ID, RUN_ID):
    job_start = time.time()
    spark_context = hopsutil._find_spark().sparkContext
    exp_driver = None
    try:
        APP_ID, RUN_ID = util.register_environment(APP_ID, RUN_ID, spark_context)
        num_executors = min(util.num_executors(spark_context), config.num_trials)

        log_dir = experiment_utils._get_logdir(APP_ID, RUN_ID)

        # Start experiment driver
        tensorboard._write_hparams_config(
            experiment_utils._get_logdir(APP_ID, RUN_ID), config.searchspace
        )

        exp_driver = OptimizationDriver(config, num_executors, log_dir)
        exp_function = exp_driver.controller.name()

        node_rdd = spark_context.parallelize(range(num_executors), num_executors)

        # Do provenance after initializing exp_driver, because exp_driver does
        # the type checks for optimizer and searchspace
        spark_context.setJobGroup(
            os.environ["ML_ID"], "{} | {}".format(config.name, exp_function)
        )

        exp_json = util.populate_experiment(config, APP_ID, RUN_ID, exp_function)
        util._log(
            "Started Maggy Experiment: {0}, {1}, run {2}".format(
                config.name, APP_ID, RUN_ID
            )
        )
        exp_driver.init(job_start)
        server_addr = exp_driver.server_addr

        # Force execution on executor, since GPU is located on executor
        worker_fct = prepare_function(
            APP_ID,
            RUN_ID,
            "optimization",
            train_fn,
            server_addr,
            config.hb_interval,
            exp_driver._secret,
            config.optimization_key,
            log_dir,
        )
        node_rdd.foreachPartition(worker_fct)
        job_end = time.time()

        result = exp_driver.finalize(job_end)
        best_logdir = (
            experiment_utils._get_logdir(APP_ID, RUN_ID) + "/" + result["best_id"]
        )

        util._finalize_experiment(
            exp_json,
            float(result["best_val"]),
            APP_ID,
            RUN_ID,
            "FINISHED",
            exp_driver.duration,
            experiment_utils._get_logdir(APP_ID, RUN_ID),
            best_logdir,
            config.optimization_key,
        )
        util._log("Finished Experiment")
        return result
    except:  # noqa: E722
        if exp_driver:
            # close logfiles of optimizer
            exp_driver.controller._close_log()
            if exp_driver.controller.pruner:
                exp_driver.controller.pruner._close_log()
            if exp_driver.exception:
                raise exp_driver.exception
        raise
    finally:
        # grace period to send last logs to sparkmagic
        # sparkmagic hb poll intervall is 5 seconds, therefore wait 6 seconds
        time.sleep(6)
        if exp_driver is not None:
            exp_driver.stop()
