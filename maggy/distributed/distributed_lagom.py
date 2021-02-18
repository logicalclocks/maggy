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
import atexit
import time


from maggy import util, tensorboard
from maggy.core.experiment_driver.DistributedDriver import DistributedDriver
from maggy.core.executors.Executor import Executor
from maggy.core.environment.singleton import EnvSing

APP_ID = None
RUNNING = False
RUN_ID = 1
EXPERIMENT_JSON = None


def distributed_lagom(train_fn, name, hb_interval, description, **kwargs):
    global RUNNING
    if RUNNING:
        raise RuntimeError("An experiment is currently running.")
    job_start = time.time()
    spark_context = util.find_spark().sparkContext
    exp_driver = None

    try:
        global APP_ID
        global EXPERIMENT_JSON
        global RUN_ID

        APP_ID, RUN_ID, RUNNING = register_environment(
            APP_ID, RUN_ID, RUNNING, spark_context
        )
        num_executors = util.num_executors(spark_context)
        exp_driver = DistributedDriver(
            name=name,
            description=description,
            num_executors=num_executors,
            hb_interval=hb_interval,
            log_dir=EnvSing.get_instance()._get_logdir(APP_ID, RUN_ID),
        )

        # Create a spark rdd partitioned into single integers, one for each executor. Allows
        # execution of functions on each executor node.
        node_rdd = spark_context.parallelize(range(num_executors), num_executors)
        spark_context.setJobGroup(
            os.environ["ML_ID"], "{0} | distributed_learning".format(name)
        )
        EXPERIMENT_JSON = EnvSing.get_instance().populate_experiment(
            name,
            "distributed_learning",
            "MAGGY",
            None,
            description,
            APP_ID,
            "max",
            "metric",
        )
        exp_ml_id = APP_ID + "_" + str(RUN_ID)
        EXPERIMENT_JSON = EnvSing.get_instance().attach_experiment_xattr(
            exp_ml_id, EXPERIMENT_JSON, "INIT"
        )
        # Initialize the experiment: Start the connection server (driver init call), prepare the
        # training function and execute it on each partition.
        util.log(
            "Started Maggy Experiment: {0}, {1}, run {2}".format(name, APP_ID, RUN_ID)
        )

        exp_driver.init(job_start)
        server_addr = exp_driver.server_addr
        # Executor wraps prepare function, patches training function. Execution on SparkExecutors is
        # triggered by foreachPartition call.
        exp_executor = Executor(exp_driver)
        worker_fct = exp_executor.prepare_function(
            APP_ID, RUN_ID, train_fn, server_addr, hb_interval, "metric", **kwargs
        )
        node_rdd.foreachPartition(worker_fct)
        print("Experiment finished.")
        return

    except Exception:  # noqa: E722
        _exception_handler(
            EnvSing.get_instance().seconds_to_milliseconds(time.time() - job_start)
        )
        if exp_driver and exp_driver.exception:
            raise exp_driver.exception
        raise
    finally:
        # grace period to send last logs to sparkmagic
        # sparkmagic hb poll intervall is 5 seconds, therefore wait 6 seconds
        time.sleep(6)
        # cleanup spark jobs
        if RUNNING and exp_driver is not None:
            exp_driver.stop()
        RUN_ID += 1
        RUNNING = False
        spark_context.setJobGroup("", "")


def register_environment(app_id, run_id, running, spark_context):
    app_id = str(spark_context.applicationId)
    app_id, run_id = util.validate_ml_id(app_id, run_id)
    running = True
    util.set_ml_id(app_id, run_id)
    # Create experiment directory.
    EnvSing.get_instance().create_experiment_dir(app_id, run_id)
    tensorboard._register(EnvSing.get_instance().get_logdir(app_id, run_id))
    return app_id, run_id, running


def _exception_handler(duration):
    """
    Handles exceptions during execution of an experiment

    :param duration: duration of the experiment until exception in milliseconds
    :type duration: int
    """
    try:
        global RUNNING
        global EXPERIMENT_JSON
        if RUNNING and EXPERIMENT_JSON is not None:
            EXPERIMENT_JSON["state"] = "FAILED"
            EXPERIMENT_JSON["duration"] = duration
            exp_ml_id = APP_ID + "_" + str(RUN_ID)
            EnvSing.get_instance().attach_experiment_xattr(
                exp_ml_id, EXPERIMENT_JSON, "FULL_UPDATE"
            )
    except Exception as err:  # pylint: disable=broad-except
        util.log(err)


def _exit_handler():
    """
    Handles jobs killed by the user.
    """
    try:
        global RUNNING
        global EXPERIMENT_JSON
        if RUNNING and EXPERIMENT_JSON is not None:
            EXPERIMENT_JSON["status"] = "KILLED"
            exp_ml_id = APP_ID + "_" + str(RUN_ID)
            EnvSing.get_instance().attach_experiment_xattr(
                exp_ml_id, EXPERIMENT_JSON, "FULL_UPDATE"
            )
    except Exception as err:  # pylint: disable=broad-except
        util.log(err)


atexit.register(_exit_handler)
