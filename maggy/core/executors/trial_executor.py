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
Module to produce the wrapper function to be executed by the executors.
"""

import builtins as __builtin__
import inspect
import json
import socket
import traceback
from typing import Callable, Any, Union

from maggy import util, tensorboard
from maggy.config import HyperparameterOptConfig, AblationConfig
from maggy.core import exceptions
from maggy.core.reporter import Reporter
from maggy.core.environment.singleton import EnvSing


def trial_executor_fn(
    train_fn: Callable,
    config: Union[HyperparameterOptConfig, AblationConfig],
    experiment_type: str,
    app_id: int,
    run_id: int,
    server_addr: str,
    hb_interval: int,
    secret: str,
    optimization_key: str,
    log_dir: str,
) -> Callable:
    """
    Wraps the user supplied training function in order to be passed to the Spark Executors.

    :param train_fn: Original training function.
    :param config: Experiment config.
    :param app_id: Maggy application ID.
    :param run_id: Maggy run ID.
    :param server_addr: IP of the Maggy worker registration RPC server.
    :param hb_interval: Worker heartbeat interval.
    :param secret: Secret string to authenticate messages.
    :param optimization key: Key of the preformance metric that should be optimized.
    :param log_dir: Location of the logger file directory on the file system.

    :returns: Patched function to execute on the Spark executors.
    """

    def _wrapper_fun(_: Any) -> None:
        """Patched function from trial_executor_fn factory.

        :param _: Necessary catch for the iterator given by Spark to the
        function upon foreach calls. Can safely be disregarded.
        """
        env = EnvSing.get_instance()

        env.set_ml_id(app_id, run_id)

        # get task context information to determine executor identifier
        partition_id, task_attempt = util.get_partition_attempt_id()

        client = EnvSing.get_instance().get_client(
            server_addr,
            partition_id,
            hb_interval,
            secret,
            socket.socket(socket.AF_INET, socket.SOCK_STREAM),
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
                if env.exists(tb_logdir):
                    util.clean_dir(tb_logdir, [trial_log_file])
                else:
                    env.mkdir(tb_logdir)

                reporter.init_logger(trial_log_file)
                tensorboard._register(tb_logdir)
                if experiment_type == "ablation":
                    env.dump(
                        json.dumps(ablation_params, default=util.json_default_numpy),
                        tb_logdir + "/.hparams.json",
                    )

                else:
                    env.dump(
                        json.dumps(parameters, default=util.json_default_numpy),
                        tb_logdir + "/.hparams.json",
                    )

                model = config.model
                dataset = config.dataset

                try:
                    reporter.log("Starting Trial: {}".format(trial_id), False)
                    reporter.log("Trial Configuration: {}".format(parameters), False)

                    if experiment_type == "optimization":
                        tensorboard._write_hparams(parameters, trial_id)

                    sig = inspect.signature(train_fn)
                    kwargs = {}
                    if sig.parameters.get("model", None):
                        kwargs["model"] = model
                    if sig.parameters.get("dataset", None):
                        kwargs["dataset"] = dataset
                    if sig.parameters.get("hparams", None):
                        kwargs["hparams"] = parameters

                    if sig.parameters.get("reporter", None):
                        kwargs["reporter"] = reporter
                        retval = train_fn(**kwargs)
                    else:
                        retval = train_fn(**kwargs)

                    # todo: test this change
                    retval_list = []
                    if isinstance(retval, dict):
                        for item in retval.items():
                            retval_list.append(item[1])
                        retval = retval_list
                    retval = {
                        "Metric": retval[0] if isinstance(retval, list) else retval
                    }
                    retval = util.handle_return_val(
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
