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
import warnings

from maggy import util
from maggy.core.rpc import Client


class BaseEnv:
    """
    Support maggy on a local pyspark installation.
    """

    def __init__(self):
        self.log_dir = os.path.join(os.getcwd(), "experiment_log")
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def set_ml_id(self, app_id=0, run_id=0):
        os.environ["ML_ID"] = str(app_id) + "_" + str(run_id)

    def create_experiment_dir(self, app_id, run_id):
        if not os.path.exists(os.path.join(self.log_dir, app_id)):
            os.mkdir(os.path.join(self.log_dir, app_id))

        experiment_path = self.get_logdir(app_id, run_id)
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)

        os.mkdir(experiment_path)

    def get_logdir(self, app_id, run_id):
        return os.path.join(self.log_dir, str(app_id), str(run_id))

    def populate_experiment(
        self,
        model_name,
        function,
        type,
        hp,
        description,
        app_id,
        direction,
        optimization_key,
    ):
        pass

    def attach_experiment_xattr(self, exp_ml_id, experiment_json, command):
        pass

    def exists(self, hdfs_path):
        return os.path.exists(hdfs_path)

    def mkdir(self, hdfs_path):
        return os.mkdir(hdfs_path)

    def isdir(self, dir_path, project=None):
        return os.path.isdir(dir_path)

    def ls(self, dir_path):
        return os.listdir(dir_path)

    def delete(self, path, recursive=False):

        if self.exists(path):
            if os.path.isdir(path):
                if recursive:
                    # remove the directory recursively
                    shutil.rmtree(path)
                elif not os.listdir(path):
                    os.rmdir(path)
                else:
                    warnings.warn(
                        "Could not delete the dir {}, not empty.\n"
                        "Use recursive=True when calling this function".format(path)
                    )
            elif os.path.isfile(path):
                os.remove(path)
        else:
            warnings.warn(
                "Could not delete the file in {}.\n"
                "File does not exists.".format(path)
            )

    def dump(self, data, hdfs_path):
        head_tail = os.path.split(hdfs_path)
        if not os.path.exists(head_tail[0]):
            os.makedirs(head_tail[0])
        with self.open_file(hdfs_path, flags="w") as file:
            file.write(data)

    def get_ip_address(self):
        sc = util.find_spark().sparkContext
        return sc._conf.get("spark.driver.host")

    def get_constants(self):
        pass

    def open_file(self, hdfs_path, flags="r", buff_size=-1):
        return open(hdfs_path, mode=flags, buffering=buff_size)

    def get_training_dataset_path(
        self, training_dataset, featurestore=None, training_dataset_version=1
    ):
        pass

    def get_training_dataset_tf_record_schema(
        self, training_dataset, training_dataset_version=1, featurestore=None
    ):
        pass

    def get_featurestore_metadata(self, featurestore=None, update_cache=False):
        pass

    def init_ml_tracking(self, app_id, run_id):
        pass

    def log_searchspace(self, app_id, run_id, searchspace):
        pass

    def connect_host(self, server_sock, server_host_port, exp_driver):
        if not server_host_port:
            server_sock.bind(("", 0))
            # hostname may not be resolvable but IP address probably will be
            host = self.get_ip_address()
            port = server_sock.getsockname()[1]
            server_host_port = (host, port)

        else:
            server_sock.bind(server_host_port)

        server_sock.listen(10)

        return server_sock, server_host_port

    def _upload_file_output(self, retval, hdfs_exec_logdir):
        pass

    def project_path(self):
        return os.getcwd()

    def get_user(self):
        return ""

    def project_name(self):
        return ""

    def finalize_experiment(
        self,
        experiment_json,
        metric,
        app_id,
        run_id,
        state,
        duration,
        logdir,
        best_logdir,
        optimization_key,
    ):
        pass

    def str_or_byte(self, str):
        return str

    def get_executors(self, sc):

        if sc._conf.get("spark.dynamicAllocation.enabled") == "true":
            maxExecutors = int(
                sc._conf.get("spark.dynamicAllocation.maxExecutors", defaultValue="-1")
            )
            if maxExecutors == -1:
                raise KeyError(
                    'Failed to find "spark.dynamicAllocation.maxExecutors" property, '
                    "but dynamicAllocation is enabled. "
                    "Define the number of min and max executors when building the spark session."
                )
        else:
            maxExecutors = int(
                sc._conf.get("spark.executor.instances", defaultValue="-1")
            )
            if maxExecutors == -1:
                raise KeyError(
                    'Failed to find "spark.executor.instances" property, '
                    'Define the number of executors using "spark.executor.instances" '
                    "when building the spark session."
                )
        return maxExecutors

    def build_summary_json(self, logdir):
        pass

    def connect_hsfs(self):
        pass

    def convert_return_file_to_arr(self, return_file):
        pass

    def upload_file_output(self, retval, hdfs_exec_logdir):
        pass

    def get_client(self, server_addr, partition_id, hb_interval, secret, sock):
        client_addr = (
            self.get_ip_address(),
            sock.getsockname()[1],
        )
        return Client(server_addr, client_addr, partition_id, 0, hb_interval, secret)
