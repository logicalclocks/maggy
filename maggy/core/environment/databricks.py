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

import os

from maggy.core.environment.base import BaseEnv
from maggy.core.rpc import Client


class DatabricksEnv(BaseEnv):
    """
    This class extends BaseEnv.
    Environment implemented for maggy usage on Databricks.
    """

    def __init__(self):
        self.log_dir = "/dbfs/maggy_log/"
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def mkdir(self, hdfs_path):
        return os.mkdir(hdfs_path)

    def project_path(self, project=None, exclude_nn_addr=False):
        return "/dbfs/"

    def get_executors(self, sc):
        if (
            sc._conf.get("spark.databricks.clusterUsageTags.clusterScalingType")
            == "autoscaling"
        ):
            maxExecutors = int(
                sc._conf.get(
                    "spark.databricks.clusterUsageTags.clusterMaxWorkers",
                    defaultValue="-1",
                )
            )
            if maxExecutors == -1:
                raise KeyError(
                    'Failed to find "spark.databricks.clusterUsageTags.clusterMaxWorkers" property, '
                    "but clusterScalingType is set to autoscaling."
                )
        else:
            maxExecutors = int(
                sc._conf.get(
                    "spark.databricks.clusterUsageTags.clusterWorkers",
                    defaultValue="-1",
                )
            )
            if maxExecutors == -1:
                raise KeyError(
                    'Failed to find "spark.databricks.clusterUsageTags.clusterWorkers" property.'
                )
        return maxExecutors

    def get_client(self, server_addr, partition_id, hb_interval, secret, sock):
        server_addr = (server_addr[0], server_addr[1])
        client_addr = (
            server_addr[0],
            sock.getsockname()[1],
        )
        return Client(server_addr, client_addr, partition_id, 0, hb_interval, secret)

    def get_logdir(self, app_id, run_id):
        return self.log_dir
