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

from hops.experiment_impl.util import experiment_utils

from maggy.core.executors import trial_executor, dist_executor
from maggy.core.experiment_driver.OptimizationDriver import OptimizationDriver
from maggy.core.experiment_driver.AblationDriver import AblationDriver
from maggy.core.experiment_driver.DistributedDriver import DistributedDriver


class Executor:
    """Wrapper class around the different executor functions.

    Executor functions monkey patch the training function to fit the experiment framework.

    Attributes:
        exp_driver (Union[OptimizationDriver, AblationDriver, DistributedDriver]): Experiment driver
            for the patching function.
    """

    def __init__(self, exp_driver):
        """Asserts if the driver is of correct and saves the driver.

        Args:
            exp_driver (Union[OptimizationDriver, AblationDriver, DistributedDriver]): Experiment
            driver for the patching function.
        """
        assert type(exp_driver) in [
            OptimizationDriver,
            AblationDriver,
            DistributedDriver,
        ], f"Experiment driver type {type(exp_driver)} unsupported by Executor."
        self.exp_driver = exp_driver

    def prepare_function(
        self,
        app_id,
        run_id,
        train_fn,
        server_addr,
        hb_interval,
        optimization_key,
        **kwargs,
    ):
        """Wrapper function for the monkey patching functions.

        Infers correct patching function from the driver type.

        Args:
            app_id (int): Maggy application ID.
            run_id (int): Maggy run ID.
            train_fn (Callable): Original training function.
            server_addr (str): IP of the Maggy worker registration RPC server.
            hb_interval (Union[float, int]): Worker heartbeat interval.
            optimization_key (str): Optimization method for hp tuning.

        Returns:
            Patched training function.
        """
        log_dir = experiment_utils._get_logdir(app_id, run_id)
        if self.exp_driver.exp_type in ["optimization", "ablation"]:
            return trial_executor._prepare_func(
                app_id,
                run_id,
                self.exp_driver.exp_type,
                train_fn,
                server_addr,
                hb_interval,
                self.exp_driver._secret,
                optimization_key,
                log_dir,
            )
        return dist_executor.prepare_function(
            app_id,
            run_id,
            train_fn,
            server_addr,
            hb_interval,
            self.exp_driver._secret,
            log_dir,
            **kwargs,
        )
