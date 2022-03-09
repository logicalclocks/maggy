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

from __future__ import annotations

import secrets
import select
import socket
import struct
import threading
import time
import typing
from typing import Any

import maggy.core.config as conf

if conf.is_spark_available():
    from pyspark import cloudpickle

from maggy.core.environment.singleton import EnvSing
from maggy.config import TfDistributedConfig
from maggy.trial import Trial

if typing.TYPE_CHECKING:  # Avoid circular import error.
    from maggy.core.experiment_driver import Driver

BUFSIZE = 1024 * 2
MAX_RETRIES = 3
SERVER_HOST_PORT = None


class Reservations(object):
    """Thread-safe store for worker reservations.

    Needs to be thread-safe mainly because the server listener thread can add
    reservations while the experiment driver might modify something on a
    reservation.
    """

    def __init__(self, required):
        """

        Args:
            required:
        """
        self.required = required
        self.lock = threading.RLock()
        self.reservations = {}
        self.check_done = False

    def add(self, meta):
        """
        Add a reservation.

        Args:
            :meta: a dictonary of metadata about a node
        """
        with self.lock:
            self.reservations[meta["partition_id"]] = {
                "host_port": meta["host_port"],
                "task_attempt": meta["task_attempt"],
                "trial_id": meta["trial_id"],
                "num_executors": self.required,
            }

            if self.remaining() == 0:
                self.check_done = True

    def done(self):
        """Returns True if the ``required`` number of reservations have been fulfilled."""
        with self.lock:
            return self.check_done

    def get(self):
        """Get the current reservations."""
        with self.lock:
            return self.reservations

    def remaining(self):
        """Get a count of remaining/unfulfilled reservations."""
        with self.lock:
            num_registered = len(self.reservations)
            return self.required - num_registered

    def get_assigned_trial(self, partition_id):
        """Get the ``trial_id`` of the trial assigned to ``partition_id``.

        Returns None if executor with ``partition_id`` is not registered or if
        ``partition_id`` is not assigned a trial yet.

        Args:
            :partition_id: An id to identify the spark executor.

        Returns:
            trial_id
        """
        with self.lock:
            reservation = self.reservations.get(partition_id, None)
            if reservation is not None:
                return reservation.get("trial_id", None)

    def assign_trial(self, partition_id, trial_id):
        """Assigns trial with ``trial_id`` to the reservation with ``partition_id``.

        Args:
            partition_id --
            trial {[type]} -- [description]
        """
        with self.lock:
            self.reservations.get(partition_id, None)["trial_id"] = trial_id


class MirroredReservations:
    """
    Thread-safe store for node reservations.
    """

    def __init__(self, required):
        self.required = required
        self.lock = threading.RLock()
        self.reservations = {"cluster": {"worker": [None] * required}}
        self.check_done = False

    def add(self, meta):
        """
        Add a reservation.

        Args:
            :meta: a dictonary of metadata about a node
        """
        with self.lock:
            self.reservations["cluster"]["worker"][meta["partition_id"]] = meta[
                "host_port"
            ]
            if self.remaining() == 0:
                # Sort the cluster_spec based on ip so adjacent workers end up on same machine
                self.reservations["cluster"]["worker"].sort(
                    key=lambda x: str(x.split(":")[0])
                )
                chief = self.reservations["cluster"]["worker"][0]
                self.reservations["cluster"]["chief"] = [chief]
                del self.reservations["cluster"]["worker"][0]
                self.check_done = True

    def done(self):
        """Returns True if the ``required`` number of reservations have been fulfilled."""
        with self.lock:
            return self.check_done

    def get(self):
        """Get the list of current reservations."""
        with self.lock:
            return self.reservations

    def remaining(self):
        """Get a count of remaining/unfulfilled reservations."""
        with self.lock:
            num_registered = 0
            for entry in self.reservations["cluster"]["worker"]:
                if entry is not None:
                    num_registered = num_registered + 1
            return self.required - num_registered

    def get_assigned_trial(self, partition_id):
        """Get the ``trial_id`` of the trial assigned to ``partition_id``.

        Returns None if executor with ``partition_id`` is not registered or if
        ``partition_id`` is not assigned a trial yet.

        Args:
            :partition_id: An id to identify the spark executor.

        Returns:
            trial_id
        """
        with self.lock:
            reservation = self.reservations.get(partition_id, None)
            if reservation is not None:
                return reservation.get("trial_id", None)

    def assign_trial(self, partition_id, trial_id):
        """Assigns trial with ``trial_id`` to the reservation with ``partition_id``.

        Args:
            partition_id --
            trial {[type]} -- [description]
        """
        with self.lock:
            self.reservations.get(partition_id, None)["trial_id"] = trial_id


class MessageSocket(object):
    """Abstract class w/ length-prefixed socket send/receive functions."""

    def receive(self, sock):
        """
        Receive a message on ``sock``

        Args:
            sock:

        Returns:

        """
        msg = None
        data = b""
        recv_done = False
        recv_len = -1
        while not recv_done:
            buf = sock.recv(BUFSIZE)
            if buf is None or len(buf) == 0:
                raise Exception("socket closed")
            if recv_len == -1:
                recv_len = struct.unpack(">I", buf[:4])[0]
                data += buf[4:]
                recv_len -= len(data)
            else:
                data += buf
                recv_len -= len(buf)
            recv_done = recv_len == 0

        if conf.is_spark_available():
            msg = cloudpickle.loads(data)
            return msg
        else:
            return data

    def send(self, sock, msg):
        """
        Send ``msg`` to destination ``sock``.

        Args:
            sock:
            msg:

        Returns:

        """
        if conf.is_spark_available():
            data = cloudpickle.dumps(msg)
        else:
            data = msg
        buf = struct.pack(">I", len(data)) + data
        sock.sendall(buf)


class Server(MessageSocket):
    """Simple socket server with length prefixed pickle messages"""

    reservations = None
    done = False

    def __init__(self, num_executors, config_class):
        """

        Args:
            num_executors:
        """
        if not num_executors > 0:
            raise ValueError("Number of executors has to be greater than zero!")
        if config_class == TfDistributedConfig:
            self.reservations = MirroredReservations(num_executors)
        else:
            self.reservations = Reservations(num_executors)

        self.callback_list = []
        self.message_callbacks = self._register_callbacks()

    def await_reservations(self, sc, status={}, timeout=600):
        """
        Block until all reservations are received.

        Args:
            sc:
            status:
            timeout:

        Returns:

        """
        timespent = 0
        while not self.reservations.done():
            print("Waiting for {} reservations.".format(self.reservations.remaining()))
            # check status flags for any errors
            if "error" in status:
                sc.cancelAllJobs()
            time.sleep(1)
            timespent += 1
            if timespent > timeout:
                raise Exception("Timed out waiting for reservations to complete")
        print("All reservations completed.")
        return self.reservations.get()

    def _handle_message(self, sock, msg, exp_driver):
        """
        Handles a  message dictionary. Expects a 'type' and 'data' attribute in
        the message dictionary.

        Args:
            sock:
            msg:

        Returns:

        """
        msg_type = msg["type"]
        resp = {}
        try:
            self.message_callbacks[msg_type](
                resp, msg, exp_driver
            )  # Prepare response in callback.
        except KeyError:
            resp["type"] = "ERR"
        MessageSocket.send(self, sock, resp)

    def _register_callbacks(self):
        message_callbacks = {}
        for key, call in self.callback_list:
            message_callbacks[key] = call
        return message_callbacks

    def start(self, exp_driver):
        """
        Start listener in a background thread.

        Returns:
            address of the Server as a tuple of (host, port)
        """
        global SERVER_HOST_PORT

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock, SERVER_HOST_PORT = EnvSing.get_instance().connect_host(
            server_sock, SERVER_HOST_PORT, exp_driver
        )

        def _listen(self, sock, driver):
            CONNECTIONS = []
            CONNECTIONS.append(sock)

            while not self.done:
                read_socks, _, _ = select.select(CONNECTIONS, [], [], 1)
                for sock in read_socks:
                    if sock == server_sock:
                        client_sock, client_addr = sock.accept()
                        CONNECTIONS.append(client_sock)
                        _ = client_addr
                    else:
                        try:
                            msg = self.receive(sock)
                            # raise exception if secret does not match
                            # so client socket gets closed
                            if not secrets.compare_digest(
                                msg["secret"], exp_driver._secret
                            ):
                                exp_driver.log(
                                    "SERVER secret: {}".format(exp_driver._secret)
                                )
                                exp_driver.log(
                                    "ERROR: wrong secret {}".format(msg["secret"])
                                )
                                raise Exception

                            self._handle_message(sock, msg, driver)
                        except Exception:
                            sock.close()
                            CONNECTIONS.remove(sock)
            server_sock.close()

        threading.Thread(
            target=_listen, args=(self, server_sock, exp_driver), daemon=True
        ).start()
        return SERVER_HOST_PORT

    def stop(self):
        """
        Stop the server's socket listener.
        """
        self.done = True


class OptimizationServer(Server):
    """Implements the server for hyperparameter optimization and ablation."""

    def __init__(self, num_executors: int, config_class):
        """Registers the callbacks for message handling.

        :param num_executors: Number of Spark executors scheduled for the
            experiment.
        """
        super().__init__(num_executors, config_class)
        self.callback_list = [
            ("REG", self._register_callback),
            ("QUERY", self._query_callback),
            ("METRIC", self._metric_callback),
            ("FINAL", self._final_callback),
            ("GET", self._get_callback),
            ("LOG", self._log_callback),
        ]
        self.message_callbacks = self._register_callbacks()

    def _register_callback(self, resp: dict, msg: dict, exp_driver: Driver) -> None:
        """Register message callback.

        Checks if the executor was registered before and reassignes lost trial,
        otherwise assignes a new trial to the executor.
        """
        lost_trial = self.reservations.get_assigned_trial(msg["partition_id"])
        if lost_trial is not None:
            # the trial or executor must have failed
            exp_driver.get_trial(lost_trial).status = Trial.ERROR
            # add a blacklist message to the worker queue
            fail_msg = {
                "partition_id": msg["partition_id"],
                "type": "BLACK",
                "trial_id": lost_trial,
            }
            self.reservations.add(msg["data"])
            exp_driver.add_message(fail_msg)
        else:
            # else add regular registration msg to queue
            self.reservations.add(msg["data"])
            exp_driver.add_message(msg)
        resp["type"] = "OK"

    def _query_callback(self, resp: dict, *_: Any) -> None:
        """Query message callback.

        Checks if all executors have been registered successfully on the server.
        """
        resp["type"] = "QUERY"
        resp["data"] = self.reservations.done()

    def _metric_callback(self, resp: dict, msg: dict, exp_driver: Driver) -> None:
        """Metric message callback.

        Determines if a trial should be stopped or not.
        """
        exp_driver.add_message(msg)
        if msg["trial_id"] is None:
            resp["type"] = "OK"
        elif msg["trial_id"] is not None and msg.get("data", None) is None:
            resp["type"] = "OK"
        else:
            # lookup executor reservation to find assigned trial
            # get early stopping flag, should be False for ablation
            flag = exp_driver.get_trial(msg["trial_id"]).get_early_stop()
            resp["type"] = "STOP" if flag else "OK"

    def _final_callback(self, resp: dict, msg: dict, exp_driver: Driver) -> None:
        """Final message callback.

        Resets the reservation to avoid sending the trial again.
        """
        self.reservations.assign_trial(msg["partition_id"], None)
        resp["type"] = "OK"
        # add metric msg to the exp driver queue
        exp_driver.add_message(msg)

    def _get_callback(self, resp: dict, msg: dict, exp_driver: Driver) -> None:
        # lookup reservation to find assigned trial
        trial_id = self.reservations.get_assigned_trial(msg["partition_id"])
        # trial_id needs to be none because experiment_done can be true but
        # the assigned trial might not be finalized yet
        if exp_driver.experiment_done and trial_id is None:
            resp["type"] = "GSTOP"
        else:
            resp["type"] = "TRIAL"
        resp["trial_id"] = trial_id
        # retrieve trial information
        if trial_id is not None:
            resp["data"] = exp_driver.get_trial(trial_id).params
            exp_driver.get_trial(trial_id).status = Trial.RUNNING
        else:
            resp["data"] = None

    def _log_callback(self, resp: dict, _: Any, exp_driver: Driver) -> None:
        """Log message callback.

        Copies logs from the driver and returns them.
        """
        # get data from experiment driver
        result, log = exp_driver.get_logs()
        resp["type"] = "OK"
        resp["ex_logs"] = log if log else None
        resp["num_trials"] = exp_driver.num_trials
        resp["to_date"] = result["num_trials"]
        resp["stopped"] = result["early_stopped"]
        resp["metric"] = result["best_val"]

    def get_assigned_trial_id(self, partition_id: int) -> dict:
        """Returns the id of the assigned trial, given a ``partition_id``.

        :param partition_id: The partition id to look up.

        :returns: The trial ID of the partition.
        """
        return self.reservations.get_assigned_trial(partition_id)


class DistributedTrainingServer(Server):
    """Implements the server for distributed training."""

    def __init__(self, num_executors: int, config_class):
        """Registers the callbacks for message handling.

        :param num_executors: Number of Spark executors scheduled for the
            experiment.
        """
        super().__init__(num_executors, config_class)
        self.callback_list = [
            ("REG", self._register_callback),
            ("METRIC", self._metric_callback),
            ("EXEC_CONFIG", self._exec_config_callback),
            ("LOG", self._log_callback),
            ("QUERY", self._query_callback),
            ("FINAL", self._final_callback),
        ]
        self.message_callbacks = self._register_callbacks()

    def _register_callback(self, resp: dict, msg: dict, exp_driver: Driver) -> None:
        """Register message callback.

        Saves workers connection metadata for initialization of distributed
        backend.
        """
        self.reservations.add(msg["data"])
        exp_driver.add_message(msg)
        resp["type"] = "OK"

    def _exec_config_callback(self, resp: dict, *_: Any) -> None:
        """Executor config message callback.

        Returns the connection info of all Spark executors registered.
        """
        try:
            resp["data"] = self.reservations.get()
        except KeyError:
            resp["data"] = None
        resp["type"] = "OK"

    def _log_callback(self, resp: dict, _: Any, exp_driver: Driver) -> None:
        """Log message callback.

        Copies logs from the driver and returns them.
        """
        _, log = exp_driver.get_logs()
        resp["type"] = "OK"
        resp["ex_logs"] = log if log else None
        resp["num_trials"] = 1
        resp["to_date"] = 0
        resp["stopped"] = False
        resp["metric"] = "N/A"

    def _metric_callback(self, resp: dict, msg: dict, exp_driver: Driver) -> None:
        """Metric message callback.

        Confirms heartbeat messages from the clients and adds logs to the driver.
        """
        exp_driver.add_message(msg)
        resp["type"] = "OK"

    def _query_callback(self, resp: dict, *_: Any) -> None:
        """Query message callback.

        Checks if all executors have been registered successfully on the server.
        """
        resp["type"] = "QUERY"
        resp["data"] = self.reservations.done()

    def _final_callback(self, resp: dict, msg: dict, exp_driver: Driver) -> None:
        """Final message callback.

        Adds final results to the message queue.
        """
        resp["type"] = "OK"
        exp_driver.add_message(msg)


class TensorflowServer(DistributedTrainingServer):
    """Implements the server for distributed training using Tensorflow."""

    def __init__(self, num_executors: int, config_class):
        """Registers the callbacks for message handling.

        :param num_executors: Number of Spark executors scheduled for the
            experiment.
        """
        super().__init__(num_executors, config_class)
        self.callback_list = [
            ("REG", self._register_callback),
            ("METRIC", self._metric_callback),
            ("TF_CONFIG", self._tf_callback),
            ("RESERVATIONS", self._get_reservations),
            ("LOG", self._log_callback),
            ("QUERY", self._query_callback),
            ("FINAL", self._final_callback),
        ]
        self.message_callbacks = self._register_callbacks()

    def _get_reservations(self, resp: dict, *_: Any) -> None:

        try:
            resp["data"] = self.reservations.get()
        except KeyError:
            resp["data"] = None
        resp["type"] = "OK"

    def _tf_callback(self, resp: dict, *_: Any) -> None:
        """Tensorflow message callback.

        Returns the connection info of the Spark worker with partition ID 1 if
        available.
        """
        try:
            # Get the config of worker with partition 1.
            resp["data"] = self.reservations.get()[0]
        except KeyError:
            resp["data"] = None
        resp["type"] = "OK"


class Client(MessageSocket):
    """BaseClient to register and await node reservations.

    Args:
        :server_addr: a tuple of (host, port) pointing to the Server.
    """

    def __init__(
        self, server_addr, client_addr, partition_id, task_attempt, hb_interval, secret
    ):
        # socket for main thread
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(server_addr)
        # socket for heartbeat thread
        self.hb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.hb_sock.connect(server_addr)
        self.server_addr = server_addr
        self.done = False
        self.client_addr = client_addr
        self.partition_id = partition_id
        self.task_attempt = task_attempt
        self.hb_interval = hb_interval
        self._secret = secret

    def _request(self, req_sock, msg_type, msg_data=None, trial_id=None, logs=None):
        """Helper function to wrap msg w/ msg_type."""
        msg = {}
        msg["partition_id"] = self.partition_id
        msg["type"] = msg_type
        msg["secret"] = self._secret

        if msg_type == "FINAL" or msg_type == "METRIC":
            msg["trial_id"] = trial_id
            if logs == "":
                msg["logs"] = None
            else:
                msg["logs"] = logs
        msg["data"] = msg_data
        done = False
        tries = 0
        while not done and tries < MAX_RETRIES:
            try:
                MessageSocket.send(self, req_sock, msg)
                done = True
            except socket.error as e:
                tries += 1
                if tries >= MAX_RETRIES:
                    raise
                print("Socket error: {}".format(e))
                req_sock.close()
                req_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                req_sock.connect(self.server_addr)
        return MessageSocket.receive(self, req_sock)

    def close(self):
        """Close the client's sockets."""
        self.sock.close()
        self.hb_sock.close()

    def register(self, registration):
        """
        Register ``registration`` with server.

        Args:
            registration:

        Returns:

        """
        resp = self._request(self.sock, "REG", registration)
        return resp

    def await_reservations(self):
        done = False
        while not done:
            done = self._request(self.sock, "QUERY").get("data", False)
            time.sleep(1)
        print("All executors registered: {}".format(done))
        return done

    def start_heartbeat(self, reporter):
        def _heartbeat(self, reporter):
            while not self.done:
                backoff = True  # Allow to tolerate HB failure on shutdown (once)
                with reporter.lock:
                    metric, step, logs = reporter.get_data()
                    data = {"value": metric, "step": step}
                    try:
                        resp = self._request(
                            self.hb_sock, "METRIC", data, reporter.get_trial_id(), logs
                        )
                    except OSError as err:  # TODO: Verify that this is necessary
                        if backoff:
                            backoff = False
                            time.sleep(5)
                            continue
                        raise OSError from err
                    self._handle_message(resp, reporter)
                time.sleep(self.hb_interval)

        threading.Thread(target=_heartbeat, args=(self, reporter), daemon=True).start()
        reporter.log("Started metric heartbeat", False)

    def get_suggestion(self, reporter):
        """Blocking call to get new parameter combination."""
        while not self.done:
            resp = self._request(self.sock, "GET")
            trial_id, parameters = self._handle_message(resp, reporter) or (None, None)

            if trial_id is not None:
                break
            time.sleep(1)
        return trial_id, parameters

    def get_message(self, msg_type, timeout=60):
        """Return the property of msg_type.

        :param msg_type: The property to request.
        :param timeout: Waiting time for the request (Default: ''60'')

        :return the property requested
        """
        config = None
        start_time = time.time()
        while not config and time.time() - start_time < timeout:
            config = self._request(self.sock, msg_type).get("data", None)
        return config

    def stop(self):
        """Stop the Clients's heartbeat thread."""
        self.done = True

    def _handle_message(self, msg, reporter=None):
        """
        Handles a  message dictionary. Expects a 'type' and 'data' attribute in
        the message dictionary.

        Args:
            sock:
            msg:

        Returns:

        """
        msg_type = msg["type"]
        # if response is STOP command, early stop the training
        if msg_type == "STOP":
            reporter.early_stop()
        elif msg_type == "GSTOP":
            reporter.log("Stopping experiment", False)
            self.done = True
        elif msg_type == "TRIAL":
            return msg["trial_id"], msg["data"]
        elif msg_type == "ERR":
            reporter.log("Stopping experiment", False)
            self.done = True

    def finalize_metric(self, metric, reporter):
        # make sure heartbeat thread can't send between sending final metric
        # and resetting the reporter
        with reporter.lock:
            _, _, logs = reporter.get_data()
            resp = self._request(
                self.sock, "FINAL", metric, reporter.get_trial_id(), logs
            )
            reporter.reset()
        return resp
