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

from maggy import util
from maggy.core.rpc import DistributedServer
from maggy.core.experiment_driver.driver import Driver
from maggy.core.executors.dist_executor import dist_executor_fn


class DistributedDriver(Driver):
    """Distributed driver class to run server in Torch registration mode."""

    def __init__(self, config, app_id, run_id):
        super().__init__(config, app_id, run_id)
        self.server = DistributedServer(self.num_executors)
        self.results = []

    def _exp_startup_callback(self):
        pass

    def _exp_final_callback(self, job_end, _):
        result = self.average_metric()
        print("Final average test loss: {:.3f}".format(result))
        print(
            "Finished experiment. Total run time: "
            + util.time_diff(self.job_start, job_end)
        )
        return result

    def _exp_exception_callback(self, exc):
        raise exc

    def _patching_fn(self, train_fn):
        return dist_executor_fn(
            train_fn,
            self.config,
            self.APP_ID,
            self.RUN_ID,
            self.server_addr,
            self.hb_interval,
            self._secret,
            self.log_dir,
        )

    def _register_msg_callbacks(self):
        self.message_callbacks["METRIC"] = self._log_msg_callback
        self.message_callbacks["FINAL"] = self._final_msg_callback

    def _log_msg_callback(self, msg):
        logs = msg.get("logs", None)
        if logs is not None:
            with self.log_lock:
                self.executor_logs = self.executor_logs + logs

    def _final_msg_callback(self, msg):
        self.results.append(msg.get("data", None))

    def average_metric(self):
        valid_results = [x for x in self.results if x is not None]
        return sum(valid_results) / len(valid_results)
