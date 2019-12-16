"""
Module to produce the wrapper function to be executed by the executors.
"""

import builtins as __builtin__
import inspect
import json

from hops import hdfs as hopshdfs
from hops.experiment_impl.util import experiment_utils

from maggy import util, tensorboard
from maggy.core import rpc, exceptions
from maggy.core.reporter import Reporter

def _prepare_func(
        app_id, run_id, experiment_type, map_fun, server_addr, hb_interval,
        secret, optimization_key, log_dir):

    def _wrapper_fun(iter):
        """
        Wraps the user supplied training function in order to be passed to the
        Spark Executors.

        Args:
            iter:

        Returns:

        """

        # get task context information to determine executor identifier
        partition_id, task_attempt = util.get_partition_attempt_id()

        client = rpc.Client(server_addr, partition_id,
                            task_attempt, hb_interval, secret)

        # save the builtin print
        original_print = __builtin__.print

        reporter = Reporter(partition_id, task_attempt, original_print)

        def maggy_print(*args, **kwargs):
            """Maggy custom print() function."""
            reporter.log(' '.join(str(x) for x in args))
            original_print(*args, **kwargs)

        # override the builtin print
        __builtin__.print = maggy_print

        try:
            client_addr = client.client_addr

            host_port = client_addr[0] + ":" + str(client_addr[1])

            exec_spec = {}
            exec_spec['partition_id'] = partition_id
            exec_spec['task_attempt'] = task_attempt
            exec_spec['host_port'] = host_port
            exec_spec['trial_id'] = None

            reporter.log("Registering with experiment driver", False)
            client.register(exec_spec)

            client.start_heartbeat(reporter)

            # blocking
            trial_id, parameters = client.get_suggestion()

            while not client.done:
                if experiment_type == 'ablation':
                    parameters.pop('ablated_feature')
                    parameters.pop('ablated_layer')

                tb_logdir = log_dir + '/' + trial_id
                log_file = tb_logdir + '/' + trial_id + '.log'
                reporter.set_trial_id(trial_id)

                # If trial is repeated, delete trial directory, except log file
                if hopshdfs.exists(tb_logdir):
                    util._clean_dir(tb_logdir, [log_file])
                else:
                    hopshdfs.mkdir(tb_logdir)

                reporter.init_logger(log_file)
                tensorboard._register(tb_logdir)
                hopshdfs.dump(json.dumps(parameters, default=util.json_default_numpy), tb_logdir + '/.hparams.json')

                try:
                    reporter.log("Starting Trial: {}".format(trial_id), False)
                    reporter.log("Trial Configuration: {}".format(parameters), False)

                    tensorboard._write_hparams(parameters)

                    sig = inspect.signature(map_fun)
                    if sig.parameters.get('reporter', None):
                        retval = map_fun(**parameters, reporter=reporter)
                    else:
                        retval = map_fun(**parameters)

                    tensorboard._write_session_end()

                    retval = util._handle_return_val(
                        retval, tb_logdir, optimization_key, log_file)

                except exceptions.EarlyStopException as e:
                    retval = e.metric
                    reporter.log("Early Stopped Trial.", False)

                client.finalize_metric(retval, reporter)
                reporter.log("Finished Trial: {}".format(trial_id), False)
                reporter.log("Final Metric: {}".format(retval), False)

                # blocking
                trial_id, parameters = client.get_suggestion()

        except:
            reporter.fd.close()
            raise
        finally:
            reporter.fd.close()
            client.stop()
            client.close()

    return _wrapper_fun
