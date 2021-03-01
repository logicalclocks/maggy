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

from maggy.core.experiment_driver.driver import Driver


class DistributedDriver(Driver):
    """Distributed driver class to run server in Torch registration mode."""

    def __init__(self, config, num_executors, log_dir):
        super().__init__(config, num_executors, log_dir)
        self.results = []
        # Legacy: Define result and num_trials to make heartbeat work.
        # Is going to get removed in the future.
        self.result = {"best_val": "n.a.", "num_trials": 1, "early_stopped": 0}
        self.num_trials = 1

    def _register_callbacks(self):
        self.message_callbacks["METRIC"] = self.log_callback
        self.message_callbacks["FINAL"] = self.final_callback

    def log_callback(self, msg):
        logs = msg.get("logs", None)
        if logs is not None:
            with self.log_lock:
                self.executor_logs = self.executor_logs + logs

    def final_callback(self, msg):
        self.results.append(msg.get("data", None))

    def average_metric(self):
        valid_results = [x for x in self.results if x is not None]
        return sum(valid_results) / len(valid_results)
