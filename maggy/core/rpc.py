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

import threading
import struct
from pyspark import cloudpickle
import time
import select
import socket
import secrets
import json

from maggy.trial import Trial

from hops import constants as hopsconstants
from hops import util as hopsutil
from hops.experiment_impl.util import experiment_utils

MAX_RETRIES = 3
BUFSIZE = 1024 * 2

server_host_port = None


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

        msg = cloudpickle.loads(data)
        return msg

    def send(self, sock, msg):
        """
        Send ``msg`` to destination ``sock``.

        Args:
            sock:
            msg:

        Returns:

        """
        data = cloudpickle.dumps(msg)
        buf = struct.pack(">I", len(data)) + data
        sock.sendall(buf)


class Server(MessageSocket):
    """Simple socket server with length prefixed pickle messages"""

    reservations = None
    done = False

    def __init__(self, count):
        """

        Args:
            count:
        """
        assert count > 0
        self.reservations = Reservations(count)

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
            print("waiting for {} reservations.".format(self.reservations.remaining()))
            # check status flags for any errors
            if "error" in status:
                sc.cancelAllJobs()
            time.sleep(1)
            timespent += 1
            if timespent > timeout:
                raise Exception("Timed out waiting for reservations to complete")
        print("All reservations completed")
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

        # Prepare message
        send = {}

        if msg_type == "REG":
            # check if executor was registered before and retrieve lost trial
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

            send["type"] = "OK"
        elif msg_type == "QUERY":
            send["type"] = "QUERY"
            send["data"] = self.reservations.done()
        elif msg_type == "METRIC":
            # add metric msg to the exp driver queue
            exp_driver.add_message(msg)

            if msg["trial_id"] is None:
                send["type"] = "OK"
                MessageSocket.send(self, sock, send)
                return
            elif msg["trial_id"] is not None:
                if msg.get("data", None) is None:
                    send["type"] = "OK"
                    MessageSocket.send(self, sock, send)
                    return

            # lookup executor reservation to find assigned trial
            trialId = msg["trial_id"]
            # get early stopping flag for hyperparameter optimization trials
            flag = False
            if exp_driver.experiment_type == "optimization":
                flag = exp_driver.get_trial(trialId).get_early_stop()

            if flag:
                send["type"] = "STOP"
            else:
                send["type"] = "OK"
        elif msg_type == "FINAL":
            # reset the reservation to avoid sending the same trial again
            self.reservations.assign_trial(msg["partition_id"], None)

            send["type"] = "OK"

            # add metric msg to the exp driver queue
            exp_driver.add_message(msg)
        elif msg_type == "GET":
            # lookup reservation to find assigned trial
            trial_id = self.reservations.get_assigned_trial(msg["partition_id"])

            # trial_id needs to be none because experiment_done can be true but
            # the assigned trial might not be finalized yet
            if exp_driver.experiment_done and trial_id is None:
                send["type"] = "GSTOP"
            else:
                send["type"] = "TRIAL"

            send["trial_id"] = trial_id

            # retrieve trial information
            if trial_id is not None:
                send["data"] = exp_driver.get_trial(trial_id).params
                exp_driver.get_trial(trial_id).status = Trial.RUNNING
            else:
                send["data"] = None
        elif msg_type == "LOG":
            # get data from experiment driver
            result, log = exp_driver._get_logs()

            send["type"] = "OK"
            if log:
                send["ex_logs"] = log
            else:
                send["ex_logs"] = None
            send["num_trials"] = exp_driver.num_trials
            send["to_date"] = result["num_trials"]
            send["stopped"] = result["early_stopped"]
            send["metric"] = result["best_val"]
        else:
            send["type"] = "ERR"

        MessageSocket.send(self, sock, send)

    def get_assigned_trial_id(self, partition_id):
        """Returns the id of the assigned trial, given a ``partition_id``.

        Arguments:
            partition_id {[type]} -- [description]

        Returns:
            trial_id
        """
        return self.reservations.get_assigned_trial(partition_id)

    def start(self, exp_driver):
        """
        Start listener in a background thread.

        Returns:
            address of the Server as a tuple of (host, port)
        """
        global server_host_port

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if not server_host_port:
            server_sock.bind(("", 0))
            # hostname may not be resolvable but IP address probably will be
            host = experiment_utils._get_ip_address()
            port = server_sock.getsockname()[1]
            server_host_port = (host, port)

            # register this driver with Hopsworks
            sc = hopsutil._find_spark().sparkContext
            app_id = str(sc.applicationId)

            method = hopsconstants.HTTP_CONFIG.HTTP_POST
            resource_url = (
                hopsconstants.DELIMITERS.SLASH_DELIMITER
                + hopsconstants.REST_CONFIG.HOPSWORKS_REST_RESOURCE
                + hopsconstants.DELIMITERS.SLASH_DELIMITER
                + "maggy"
                + hopsconstants.DELIMITERS.SLASH_DELIMITER
                + "drivers"
            )
            json_contents = {
                "hostIp": host,
                "port": port,
                "appId": app_id,
                "secret": exp_driver._secret,
            }
            json_embeddable = json.dumps(json_contents)
            headers = {
                hopsconstants.HTTP_CONFIG.HTTP_CONTENT_TYPE: hopsconstants.HTTP_CONFIG.HTTP_APPLICATION_JSON
            }

            try:
                response = hopsutil.send_request(
                    method, resource_url, data=json_embeddable, headers=headers
                )

                if (response.status_code // 100) != 2:
                    print("No connection to Hopsworks for logging.")
                    exp_driver._log("No connection to Hopsworks for logging.")
            except Exception as e:
                print("Connection failed to Hopsworks. No logging.")
                exp_driver._log(e)
                exp_driver._log("Connection failed to Hopsworks. No logging.")
        else:
            server_sock.bind(server_host_port)
        server_sock.listen(10)

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
                                exp_driver._log(
                                    "SERVER secret: {}".format(exp_driver._secret)
                                )
                                exp_driver._log(
                                    "ERROR: wrong secret {}".format(msg["secret"])
                                )
                                raise Exception

                            self._handle_message(sock, msg, driver)
                        except Exception as e:
                            _ = e
                            sock.close()
                            CONNECTIONS.remove(sock)

            server_sock.close()

        t = threading.Thread(target=_listen, args=(self, server_sock, exp_driver))
        t.daemon = True
        t.start()

        return server_host_port

    def stop(self):
        """
        Stop the server's socket listener.
        """
        self.done = True


class Client(MessageSocket):
    """Client to register and await node reservations.

    Args:
        :server_addr: a tuple of (host, port) pointing to the Server.
    """

    def __init__(self, server_addr, partition_id, task_attempt, hb_interval, secret):
        # socket for main thread
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(server_addr)
        # socket for heartbeat thread
        self.hb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.hb_sock.connect(server_addr)
        self.server_addr = server_addr
        self.done = False
        self.client_addr = (
            experiment_utils._get_ip_address(),
            self.sock.getsockname()[1],
        )
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

        # if msg_data or ((msg_data == True) or (msg_data == False)):
        #    msg['data'] = msg_data
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

        resp = MessageSocket.receive(self, req_sock)

        return resp

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
        def _heartbeat(self, report):

            while not self.done:

                with report.lock:
                    metric, step, logs = report.get_data()
                    data = {"value": metric, "step": step}

                    resp = self._request(
                        self.hb_sock, "METRIC", data, report.get_trial_id(), logs
                    )
                    _ = self._handle_message(resp, report)

                # sleep one second
                time.sleep(self.hb_interval)

        t = threading.Thread(target=_heartbeat, args=(self, reporter))
        t.daemon = True
        t.start()

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
