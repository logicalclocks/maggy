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

from abc import ABC, abstractmethod


class AbstractEnv(ABC):
    """
    Abstract class for environment definition.
    Define all the methods within this class to use a custom environment.
    """

    @abstractmethod
    def set_ml_id(self, app_id, run_id):
        pass

    @abstractmethod
    def create_experiment_dir(self, app_id, run_id):
        pass

    @abstractmethod
    def get_logdir(self, app_id, run_id):
        pass

    @abstractmethod
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

    @abstractmethod
    def attach_experiment_xattr(self, exp_ml_id, experiment_json, command):
        pass

    @abstractmethod
    def exists(self, hdfs_path, project=None):
        pass

    @abstractmethod
    def mkdir(self, hdfs_path, project=None):
        pass

    @abstractmethod
    def dump(self, data, hdfs_path):
        pass

    @abstractmethod
    def get_ip_address(self):
        pass

    @abstractmethod
    def get_constants(self):
        pass

    @abstractmethod
    def open_file(self, hdfs_path, project=None, flags="r", buff_size=0):
        pass

    @abstractmethod
    def get_training_dataset_path(
        self, training_dataset, featurestore=None, training_dataset_version=1
    ):
        pass

    @abstractmethod
    def get_training_dataset_tf_record_schema(
        self, training_dataset, training_dataset_version=1, featurestore=None
    ):
        pass

    @abstractmethod
    def get_featurestore_metadata(self, featurestore=None, update_cache=False):
        pass

    @abstractmethod
    def init_ml_tracking(self, app_id, run_id):
        pass

    @abstractmethod
    def log_searchspace(self, app_id, run_id, searchspace):
        pass

    @abstractmethod
    def connect_host(self, server_sock, server_host_port, exp_driver):
        pass

    @abstractmethod
    def isdir(self, dir_path, project=None):
        pass

    @abstractmethod
    def ls(self, dir_path, recursive=False, project=None):
        pass

    @abstractmethod
    def delete(self, path, recursive=False):
        pass

    @abstractmethod
    def upload_file_output(self, retval, hdfs_exec_logdir):
        pass

    @abstractmethod
    def project_path(self, project=None, exclude_nn_addr=False):
        pass

    @abstractmethod
    def get_user(self):
        pass

    @abstractmethod
    def project_name(self):
        pass

    @abstractmethod
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

    @abstractmethod
    def str_or_byte(self, str):
        pass

    @abstractmethod
    def get_executors(self, sc):
        pass

    @abstractmethod
    def build_summary_json(self, logdir):
        pass

    @abstractmethod
    def convert_return_file_to_arr(self, return_file):
        pass

    @abstractmethod
    def connect_hsfs(self, engine="training"):
        pass
