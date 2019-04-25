
import threading
import struct
import pickle
import time
import select
import socket

from maggy import util
from maggy.trial import Trial

MAX_RETRIES = 3
BUFSIZE = 1024 * 2


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
            self.reservations[meta['partition_id']] = {
                'host_port': meta["host_port"],
                'task_attempt': meta['task_attempt'],
                'trial_id': meta['trial_id']}

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
                return reservation.get('trial_id', None)

    def assign_trial(self, partition_id, trial_id):
        """Assigns trial with ``trial_id`` to the reservation with ``partition_id``.

        Args:
            partition_id --
            trial {[type]} -- [description]
        """
        with self.lock:
            self.reservations.get(partition_id, None)['trial_id'] = trial_id


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
        data = b''
        recv_done = False
        recv_len = -1
        while not recv_done:
            buf = sock.recv(BUFSIZE)
            if buf is None or len(buf) == 0:
                raise Exception("socket closed")
            if recv_len == -1:
                recv_len = struct.unpack('>I', buf[:4])[0]
                data += buf[4:]
                recv_len -= len(data)
            else:
                data += buf
                recv_len -= len(buf)
            recv_done = (recv_len == 0)

        msg = pickle.loads(data)
        return msg

    def send(self, sock, msg):
        """
        Send ``msg`` to destination ``sock``.

        Args:
            sock:
            msg:

        Returns:

        """
        data = pickle.dumps(msg)
        buf = struct.pack('>I', len(data)) + data
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
            print("waiting for {} reservations."
                  .format(self.reservations.remaining()))
            # check status flags for any errors
            if 'error' in status:
                sc.cancelAllJobs()
            time.sleep(1)
            timespent += 1
            if (timespent > timeout):
                raise Exception(
                    "Timed out waiting for reservations to complete")
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
        msg_type = msg['type']

        if 'trial_id' in msg:
            # throw away idle heartbeat messages without trial
            if msg['trial_id'] is None:
                send = {}
                send['type'] = 'OK'
                MessageSocket.send(self, sock, send)
                return
            # it can happen that executor has trial but no metric was reported
            # yet so trial_id is not None but data is None
            elif msg['trial_id'] is not None:
                if msg.get('data', None) is None:
                    send = {}
                    send['type'] = 'OK'
                    MessageSocket.send(self, sock, send)
                    return

        if msg_type == 'REG':
            # check if executor was registered before and retrieve lost trial
            lost_trial = self.reservations.get_assigned_trial(msg['partition_id'])
            if lost_trial is not None:
                # the trial or executor must have failed
                exp_driver.get_trial(lost_trial).status = Trial.ERROR
                # add a blacklist message to the worker queue
                fail_msg = {'partition_id': msg['partition_id'],
                    'type': 'BLACK',
                    'trial_id': lost_trial}
                self.reservations.add(msg['data'])
                exp_driver.add_message(fail_msg)
            else:
                # else add regular registration msg to queue
                self.reservations.add(msg['data'])
                exp_driver.add_message(msg)

            send = {}
            send['type'] = 'OK'

            MessageSocket.send(self, sock, send)
        elif msg_type == 'QUERY':

            send = {}
            send['type'] = 'QUERY'
            send['data'] = self.reservations.done()

            MessageSocket.send(self, sock, send)
        elif msg_type == 'METRIC':
            # Prepare message
            send = {}

            # lookup executor reservation to find assigned trial
            trialId = msg['trial_id']
            # get early stopping flag
            flag = exp_driver.get_trial(trialId).get_early_stop()
            # add metric msg to the exp driver queue
            exp_driver.add_message(msg)

            if flag:
                send['type'] = 'STOP'
            else:
                send['type'] = 'OK'

            MessageSocket.send(self, sock, send)
        elif msg_type == 'FINAL':
            # reset the reservation to avoid sending the same trial again
            self.reservations.assign_trial(msg['partition_id'], None)

            send = {}
            send['type'] = 'OK'

            # add metric msg to the exp driver queue
            exp_driver.add_message(msg)

            MessageSocket.send(self, sock, send)
        elif msg_type == 'GET':
            # lookup reservation to find assigned trial
            trial_id = self.reservations.get_assigned_trial(msg['partition_id'])

            # trial_id needs to be none because experiment_done can be true but
            # the assigned trial might not be finalized yet
            if exp_driver.experiment_done and trial_id is None:
                send = {}
                send['type'] = "GSTOP"
            else:
                send = {}
                send['type'] = "TRIAL"

            send['trial_id'] = trial_id

            # retrieve trial information
            if trial_id is not None:
                send['data'] = exp_driver.get_trial(trial_id).params
                exp_driver.get_trial(trial_id).status = Trial.RUNNING
            else:
                send['data'] = None

            MessageSocket.send(self, sock, send)
        else:
            # Prepare message
            send = {}
            send['type'] = "ERR"
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
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('', 0))
        server_sock.listen(10)

        # hostname may not be resolvable but IP address probably will be
        host = util._get_ip_address()
        port = server_sock.getsockname()[1]
        addr = (host, port)

        def _listen(self, sock, driver):
            CONNECTIONS = []
            CONNECTIONS.append(sock)

            while not self.done:
                read_socks, _, _ = select.select(
                    CONNECTIONS, [], [], 60)
                for sock in read_socks:
                    if sock == server_sock:
                        client_sock, client_addr = sock.accept()
                        CONNECTIONS.append(client_sock)
                        _ = client_addr
                    else:
                        try:
                            msg = self.receive(sock)
                            self._handle_message(sock, msg, driver)
                        except Exception as e:
                            _ = e
                            sock.close()
                            CONNECTIONS.remove(sock)

            server_sock.close()

        t = threading.Thread(target=_listen, args=(self, server_sock, exp_driver))
        t.daemon = True
        t.start()

        return addr

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
    def __init__(self, server_addr, partition_id, task_attempt, hb_interval):
        # socket for main thread
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(server_addr)
        # socket for heartbeat thread
        self.hb_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.hb_sock.connect(server_addr)
        self.server_addr = server_addr
        self.done = False
        self.client_addr = (util._get_ip_address(), self.sock.getsockname()[1])
        self.partition_id = partition_id
        self.task_attempt = task_attempt
        self.hb_interval = hb_interval

    def _request(self, req_sock, msg_type, msg_data=None, trial_id=None):
        """Helper function to wrap msg w/ msg_type."""
        msg = {}
        msg['partition_id'] = self.partition_id
        msg['type'] = msg_type

        if msg_type == 'FINAL' or msg_type == 'METRIC':
            msg['trial_id'] = trial_id

        if msg_data or ((msg_data == True) or (msg_data == False)):
            msg['data'] = msg_data

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
        resp = self._request(self.sock, 'REG', registration)
        return resp

    def await_reservations(self):
        done = False
        while not done:
            done = self._request(self.sock, 'QUERY').get('data', False)
            time.sleep(1)
        print("All executors registered: {}".format(done))
        return done

    def start_heartbeat(self, reporter):

        def _heartbeat(self, report):

            while not self.done:

                metric = report.get_metric()

                resp = self._request(self.hb_sock,
                                    'METRIC',
                                     metric,
                                     report.get_trial_id())
                _ = self._handle_message(resp, report)

                # sleep one second
                time.sleep(self.hb_interval)

        t = threading.Thread(target=_heartbeat, args=(self, reporter))
        t.daemon = True
        t.start()

        print("Started metric heartbeat")

    def get_suggestion(self):
        """Blocking call to get new parameter combination."""
        while not self.done:
            resp = self._request(self.sock, 'GET')
            trial_id, parameters = self._handle_message(resp) or (None, None)

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
        msg_type = msg['type']
        # if response is STOP command, early stop the training
        if msg_type == 'STOP':
            reporter.early_stop()
        elif msg_type == 'GSTOP':
            print("Stopping experiment")
            self.done = True
        elif msg_type == 'TRIAL':
            return msg['trial_id'], msg['data']
        elif msg_type == 'ERR':
            print("Stopping experiment")
            self.done = True

    def finalize_metric(self, metric, reporter):
        # make sure heartbeat thread can't send between sending final metric
        # and resetting the reporter
        with reporter.lock:
            resp = self._request(self.sock, 'FINAL', metric, reporter.get_trial_id())
            reporter.reset()
        return resp
