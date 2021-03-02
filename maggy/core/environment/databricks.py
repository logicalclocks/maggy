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
import shutil

from maggy import util
from maggy.core.environment.base import BaseEnv


class DatabricksEnv(BaseEnv):
    """
    Environment implemented for maggy usage on Databricks.
    """

    def __init__(self):
        self.log_dir = "/dbfs/maggy_log/"
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.constants = []

    def set_ml_id(self, app_id = 0, run_id = 0):
        os.environ['ML_ID'] = str(app_id) + '_' + str(run_id)

    def create_experiment_dir(self, app_id, run_id):
        if not os.path.exists(os.path.join(self.log_dir, app_id)):
            os.mkdir(os.path.join(self.log_dir, app_id))

        experiment_path = self.get_logdir(app_id, run_id)
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)

        os.mkdir(experiment_path)

    def mkdir(self, hdfs_path):
        return os.mkdir(hdfs_path)

    def dump(self, data, hdfs_path):
        head_tail = os.path.split(hdfs_path)
        if not os.path.exists(head_tail[0]):
            os.mkdir(head_tail[0])
        file = self.open_file(hdfs_path, flags='w')
        file.write(data)

    def get_ip_address(self):
        sc = util.find_spark().sparkContext
        return sc._conf.get("spark.driver.host")

    def delete(self, path, recursive=False):
        if self.exists(path):
            if os.path.isdir(path):
                os.rmdir(path)
            elif os.path.isfile(path):
                os.remove(path)

    def project_path(self, project=None, exclude_nn_addr=False):
        return "/dbfs/"

    def get_executors(self, sc):
        try:
            if sc._conf.get("spark.databricks.clusterUsageTags.clusterScalingType") == "autoscaling":
                maxExecutors = int(sc._conf.get("spark.databricks.clusterUsageTags.clusterMaxWorkers"))
            else:
                maxExecutors = int(sc._conf.get("spark.databricks.clusterUsageTags.clusterWorkers"))

            return maxExecutors
        except:  # noqa: E722
            raise RuntimeError(
                "Failed to find some of the spark.databricks properties."
            )
            
