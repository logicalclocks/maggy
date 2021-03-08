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
The experiment driver implements the functionality for scheduling trials on
maggy.
"""
import queue
import threading
import secrets
from abc import ABC, abstractmethod
from datetime import datetime

from maggy.core.rpc import Server
from maggy.core.environment.singleton import EnvSing


DRIVER_SECRET = None


class Driver(ABC):

    SECRET_BYTES = 8

    def __init__(self, config, num_executors, log_dir):
        global DRIVER_SECRET
        self.name = config.name
        self.description = config.description
        self.num_executors = num_executors
        self.hb_interval = config.hb_interval
        self.server = Server(self.num_executors)
        self.server_addr = None
        self.job_start = None
        DRIVER_SECRET = (
            DRIVER_SECRET if DRIVER_SECRET else self._generate_secret(self.SECRET_BYTES)
        )
        self._secret = DRIVER_SECRET
        # Logging related initialization
        self._message_q = queue.Queue()
        self.message_callbacks = {}
        self._register_callbacks()
        self.worker_done = False
        self.executor_logs = ""
        self.log_lock = threading.RLock()
        log_file = log_dir + "/maggy.log"
        self.log_dir = log_dir
        # Open File desc for HDFS to log
        if not EnvSing.get_instance().exists(log_file):
            EnvSing.get_instance().dump("", log_file)
        self.log_file_handle = EnvSing.get_instance().open_file(log_file, flags="w")
        self.exception = None
        self.result = None

    @staticmethod
    def _generate_secret(nbytes):
        """Generates a secret to be used by all clients during the experiment
        to authenticate their messages with the experiment driver.
        """
        return secrets.token_hex(nbytes=nbytes)

    def init(self, job_start):
        self.server_addr = self.server.start(self)
        self.job_start = job_start
        self._start_worker()

    def _start_worker(self):
        def _digest_queue(self):
            try:
                while not self.worker_done:
                    try:
                        msg = self._message_q.get_nowait()
                    except queue.Empty:
                        msg = {"type": None}
                    if msg["type"] in self.message_callbacks.keys():
                        self.message_callbacks[msg["type"]](
                            msg
                        )  # Execute registered callbacks.
            except Exception as exc:  # pylint: disable=broad-except
                self.log(exc)
                self.exception = exc
                self.server.stop()
                raise

        threading.Thread(target=_digest_queue, args=(self,), daemon=True).start()

    @abstractmethod
    def _register_callbacks(self):
        pass

    def add_message(self, msg):
        self._message_q.put(msg)

    def get_logs(self):
        """Return current experiment status and executor logs to send them to
        spark magic.
        """
        with self.log_lock:
            temp = self.executor_logs
            # clear the executor logs since they are being sent
            self.executor_logs = ""
            return self.result, temp

    def stop(self):
        """Stop the Driver's worker thread and server."""
        self.worker_done = True
        self.server.stop()
        self.log_file_handle.flush()
        self.log_file_handle.close()

    def log(self, log_msg):
        """Logs a string to the maggy driver log file.
        """
        msg = datetime.now().isoformat() + ": " + str(log_msg)
        self.log_file_handle.write(EnvSing.get_instance().str_or_byte((msg + "\n")))
