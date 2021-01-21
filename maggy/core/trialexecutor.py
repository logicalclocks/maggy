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
Module to produce the wrapper function to be executed by the executors.
"""

import builtins as __builtin__
import inspect
import json
import traceback

from hops import hdfs as hopshdfs
from hops.experiment_impl.util import experiment_utils

from maggy import util, tensorboard
from maggy.core import rpc, exceptions
from maggy.core.reporter import Reporter


def _prepare_func(
    app_id,
    run_id,
    experiment_type,
    train_fn,
    server_addr,
    hb_interval,
    secret,
    optimization_key,
    log_dir,
):
    def _wrapper_fun(iter):
        """
        Wraps the user supplied training function in order to be passed to the
        Spark Executors.

        Args:
            iter:

        Returns:

        """
        experiment_utils._set_ml_id(app_id, run_id)

        # get task context information to determine executor identifier
        partition_id, task_attempt = util.get_partition_attempt_id()

        client = rpc.Client(
            server_addr, partition_id, task_attempt, hb_interval, secret
        )
        log_file = (
            log_dir
            + "/executor_"
            + str(partition_id)
            + "_"
            + str(task_attempt)
            + ".log"
        )

        # save the builtin print
        original_print = __builtin__.print

        reporter = Reporter(log_file, partition_id, task_attempt, original_print)

        def maggy_print(*args, **kwargs):
            """Maggy custom print() function."""
            original_print(*args, **kwargs)
            reporter.log(" ".join(str(x) for x in args), True)

        # override the builtin print
        __builtin__.print = maggy_print

        try:
            client_addr = client.client_addr

            host_port = client_addr[0] + ":" + str(client_addr[1])

            exec_spec = {}
            exec_spec["partition_id"] = partition_id
            exec_spec["task_attempt"] = task_attempt
            exec_spec["host_port"] = host_port
            exec_spec["trial_id"] = None

            reporter.log("Registering with experiment driver", False)
            client.register(exec_spec)

            client.start_heartbeat(reporter)

            # blocking
            trial_id, parameters = client.get_suggestion(reporter)

            while not client.done:
                if experiment_type == "ablation":
                    ablation_params = {
                        "ablated_feature": parameters.get("ablated_feature", "None"),
                        "ablated_layer": parameters.get("ablated_layer", "None"),
                    }
                    parameters.pop("ablated_feature")
                    parameters.pop("ablated_layer")

                tb_logdir = log_dir + "/" + trial_id
                trial_log_file = tb_logdir + "/output.log"
                reporter.set_trial_id(trial_id)

                # If trial is repeated, delete trial directory, except log file
                if hopshdfs.exists(tb_logdir):
                    util._clean_dir(tb_logdir, [trial_log_file])
                else:
                    hopshdfs.mkdir(tb_logdir)

                reporter.init_logger(trial_log_file)
                tensorboard._register(tb_logdir)
                if experiment_type == "ablation":
                    hopshdfs.dump(
                        json.dumps(ablation_params, default=util.json_default_numpy),
                        tb_logdir + "/.hparams.json",
                    )

                else:
                    hopshdfs.dump(
                        json.dumps(parameters, default=util.json_default_numpy),
                        tb_logdir + "/.hparams.json",
                    )

                try:
                    reporter.log("Starting Trial: {}".format(trial_id), False)
                    reporter.log("Trial Configuration: {}".format(parameters), False)

                    if experiment_type == "optimization":
                        tensorboard._write_hparams(parameters, trial_id)

                    sig = inspect.signature(train_fn)
                    if sig.parameters.get("reporter", None):
                        retval = train_fn(**parameters, reporter=reporter)
                    else:
                        retval = train_fn(**parameters)

                    retval = util._handle_return_val(
                        retval, tb_logdir, optimization_key, trial_log_file
                    )

                except exceptions.EarlyStopException as e:
                    retval = e.metric
                    reporter.log("Early Stopped Trial.", False)

                reporter.log("Finished Trial: {}".format(trial_id), False)
                reporter.log("Final Metric: {}".format(retval), False)
                client.finalize_metric(retval, reporter)

                # blocking
                trial_id, parameters = client.get_suggestion(reporter)

        except:  # noqa: E722
            reporter.log(traceback.format_exc(), False)
            raise
        finally:
            reporter.close_logger()
            client.stop()
            client.close()

    return _wrapper_fun
