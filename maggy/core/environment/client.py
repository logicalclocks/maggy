import socket
import threading
import time

from maggy.core.rpc import MessageSocket

MAX_RETRIES = 3


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
