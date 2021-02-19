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

from hops.experiment_impl.util import experiment_utils

from maggy import util
from maggy.core.lagom import utils as lagom_utils
from maggy.core.experiment_driver.DistributedDriver import DistributedDriver
from maggy.core.executors.dist_executor import prepare_function


def lagom_distributed(train_fn, config, APP_ID, RUN_ID):
    job_start = time.time()
    spark_context = util.find_spark().sparkContext
    exp_driver = None
    try:
        APP_ID, RUN_ID = lagom_utils.register_environment(APP_ID, RUN_ID, spark_context)
        num_executors = util.num_executors(spark_context)
        log_dir = experiment_utils._get_logdir(APP_ID, RUN_ID)
        exp_driver = DistributedDriver(config, num_executors, log_dir)

        # Create a spark rdd partitioned into single integers, one for each executor. Allows
        # execution of functions on each executor node.
        node_rdd = spark_context.parallelize(range(num_executors), num_executors)
        spark_context.setJobGroup(
            os.environ["ML_ID"], "{0} | torch_ddp".format(config.name)
        )
        lagom_utils.populate_experiment(config, APP_ID, RUN_ID)
        # Initialize the experiment: Start the connection server (driver init call), prepare the
        # training function and execute it on each partition.
        util._log(
            "Started Maggy Experiment: {0}, {1}, run {2}".format(
                config.name, APP_ID, RUN_ID
            )
        )
        exp_driver.init(job_start)
        server_addr = exp_driver.server_addr

        # Executor wraps prepare function, patches training function. Execution on SparkExecutors is
        # triggered by foreachPartition call.
        worker_fct = prepare_function(
            APP_ID,
            RUN_ID,
            train_fn,
            config.model,
            config.train_set,
            config.test_set,
            server_addr,
            config.hb_interval,
            exp_driver._secret,
            log_dir,
        )
        node_rdd.foreachPartition(worker_fct)
        job_end = time.time()
        util._log("Final average test loss: {:.3f}".format(exp_driver.average_metric()))
        total_time = job_end - job_start
        mon, sec = divmod(total_time, 60)
        hour, mon = divmod(mon, 60)
        util._log(
            "Total training time: {:.0f} h, {:.0f} min, {:.0f} s".format(hour, mon, sec)
        )
    finally:
        time.sleep(6)
        if exp_driver:
            exp_driver.stop()
