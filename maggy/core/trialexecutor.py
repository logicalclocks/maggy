import socket
import time
from maggy import util
from maggy.core import rpc
from maggy.core.reporter import Reporter
from pyspark import TaskContext


def _prepare_func(app_id, run_id, map_fun, server_addr, hb_interval):

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
                            task_attempt, hb_interval)
        reporter = Reporter()

        try:
            client_addr = client.client_addr

            host_port = client_addr[0] + ":" + str(client_addr[1])

            exec_spec = {}
            exec_spec['partition_id'] = partition_id
            exec_spec['task_attempt'] = task_attempt
            exec_spec['host_port'] = host_port
            exec_spec['trial_id'] = None

            print("Registering with experiment driver")
            client.register(exec_spec)

            # blocking
            # _ = client.await_reservations()

            client.start_heartbeat(reporter)

            # blocking
            trial_id, parameters = client.get_suggestion()

            while not client.done:

                reporter.set_trial_id(trial_id)

                try:
                    print("--------------------------------")
                    print("Starting Trial: {}".format(trial_id))
                    print("Parameter Combination: {}".format(parameters))
                    retval = map_fun(**parameters, reporter=reporter)
                except util.EarlyStopException as e:
                    retval = e.metric
                    print("Early Stopped Trial.")
                finally:
                    client.finalize_metric(retval, reporter)
                    print("Finished Trial: {}".format(trial_id))
                    print("Final Metric: {}".format(retval))
                    print("--------------------------------\n")

                # blocking
                trial_id, parameters = client.get_suggestion()

        except:
            raise
        finally:
            client.stop()
            client.close()

    return _wrapper_fun
