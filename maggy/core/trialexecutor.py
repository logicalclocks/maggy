import builtins as __builtin__

import socket
import time
import inspect
import json

from maggy import util, tensorboard, constants, trial
from maggy.core import rpc, exceptions, config
from maggy.core.reporter import Reporter
from pyspark import TaskContext

from hops import hdfs as hopshdfs
from hops.experiment_impl.util import experiment_utils

__import__("tensorflow").compat.v1.enable_eager_execution()
import tensorflow.compat.v2 as tf
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import summary
from tensorboard.plugins.hparams import summary_v2


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
        log_file = log_dir + '/executor_' + str(partition_id) + '_' + str(task_attempt) + '.log'

        # save the builtin print
        original_print = __builtin__.print

        reporter = Reporter(log_file, partition_id, task_attempt, original_print)

        def maggy_print(*args, **kwargs):
            """Maggy custom print() function."""
            reporter.log(' '.join(str(x) for x in args))
            original_print(*args, **kwargs)

        # override the builtin print
        __builtin__.print = maggy_print

        # override HParams id with our trial id
        # 1.15.0 will have functionality to pass custom id
        summary_v2._derive_session_group_name = trial.Trial._generate_id

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

            # blocking
            # _ = client.await_reservations()

            client.start_heartbeat(reporter)

            # blocking
            # XXX separate suggestion calls for different types?
            trial_id, parameters = client.get_suggestion()

            while not client.done:
                if experiment_type == 'ablation':
                    parameters.pop('ablated_feature')
                    parameters.pop('ablated_layer')

                reporter.set_trial_id(trial_id)

                tb_logdir = log_dir + '/' + trial_id

                # If trial is repeated, delete trial directory
                if hopshdfs.exists(tb_logdir):
                    hopshdfs.delete(tb_logdir, recursive=True)

                hopshdfs.mkdir(tb_logdir)
                tensorboard._register(tb_logdir)
                hopshdfs.dump(json.dumps(parameters, default=util.json_default_numpy), tb_logdir + '/.hparams.json')

                _writer = tf.summary.create_file_writer(tb_logdir)

                try:
                    reporter.log("Starting Trial: {}".format(trial_id), False)
                    reporter.log("Trial Configuration: {}".format(parameters), False)

                    with _writer.as_default():
                        summary_v2.hparams(parameters)

                    sig = inspect.signature(map_fun)
                    if sig.parameters.get('reporter', None):
                        #TODO if killed by user, it might be that retval is not assigned and experiments will show running
                        retval = map_fun(**parameters, reporter=reporter)
                    else:
                        retval = map_fun(**parameters)

                    with _writer.as_default():
                        pb = summary.session_end_pb(api_pb2.STATUS_SUCCESS)
                        raw_pb = pb.SerializeToString()
                        tf.summary.experimental.write_raw_pb(raw_pb, step=0)
                    _writer = None

                    experiment_utils._handle_return(retval, tb_logdir, optimization_key)

                    # Make sure user function returns a numeric value
                    #if retval is None:
                    #    reporter.log(
                    #        "ERROR: Training function can't return None", False)
                    #    raise Exception("Training function can't return None")
                    #elif not isinstance(retval, constants.USER_FCT.RETURN_TYPES):
                    #    reporter.log(
                    #        "ERROR: Training function returns non numeric value: {}"
                    #        .format(type(retval)), False)
                    #    raise Exception(
                    #        "ERROR: Training function returns non numeric value: {}"
                    #        .format(type(retval)))

                except exceptions.EarlyStopException as e:
                    retval = e.metric
                    reporter.log("Early Stopped Trial.", False)
                finally:
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
