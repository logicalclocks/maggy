import socket
import time
from maggy import util, tensorboard
from maggy.core import rpc, exceptions, config
from maggy.core.reporter import Reporter
from pyspark import TaskContext

if config.mode is config.HOPSWORKS:
    import hops.util as hopsutil
    import hops.hdfs as hdfs
    import tensorflow as tf
    from tensorboard.plugins.hparams import api_pb2
    from tensorboard.plugins.hparams import summary
    from tensorboard.plugins.hparams import summary_v2


def _prepare_func(app_id, run_id, map_fun, server_addr, hb_interval, secret, app_dir):

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
        log_file = app_dir + '/logs/executor_' + str(partition_id) + '_' + str(task_attempt)
        reporter = Reporter(log_file)

        try:
            client_addr = client.client_addr

            host_port = client_addr[0] + ":" + str(client_addr[1])

            exec_spec = {}
            exec_spec['partition_id'] = partition_id
            exec_spec['task_attempt'] = task_attempt
            exec_spec['host_port'] = host_port
            exec_spec['trial_id'] = None

            reporter.log("Registering with experiment driver", True)
            client.register(exec_spec)

            # blocking
            # _ = client.await_reservations()

            client.start_heartbeat(reporter)

            # blocking
            trial_id, parameters = client.get_suggestion()

            while not client.done:

                reporter.set_trial_id(trial_id)

                if config.mode is config.HOPSWORKS:
                    logdir = app_dir + '/trials/' + trial_id
                    tensorboard._register(logdir)
                    hdfs.mkdir(logdir)

                try:
                    reporter.log("Starting Trial: {}".format(trial_id), True)
                    reporter.log("Parameter Combination: {}".format(parameters), True)
                    retval = map_fun(**parameters, reporter=reporter)
                except exceptions.EarlyStopException as e:
                    retval = e.metric
                    reporter.log("Early Stopped Trial.", True)
                finally:
                    client.finalize_metric(retval, reporter)
                    reporter.log("Finished Trial: {}".format(trial_id), True)
                    reporter.log("Final Metric: {}".format(retval), True)

                # blocking
                trial_id, parameters = client.get_suggestion()

        except:
            if config.mode is config.HOPSWORKS:
                reporter.fd.close()
            raise
        finally:
            if config.mode is config.HOPSWORKS:
                reporter.fd.close()
            client.stop()
            client.close()

    return _wrapper_fun
