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

import json
import os

import hsfs
from hops import constants as hopsconstants
from hops import hdfs as hopshdfs
from hops import util as hopsutil
from hops.experiment_impl.util import experiment_utils

from maggy import tensorboard
from maggy import util
from maggy.core.environment.base import BaseEnv

global APP_ID


class HopsworksEnv(BaseEnv):
    """
    This class extends BaseEnv.
    The methods implemented mainly recall the libraries developed on maggy.
    """

    APP_ID = None

    def set_ml_id(self, app_id=0, run_id=0):
        return experiment_utils._set_ml_id(app_id, run_id)

    def set_app_id(self, app_id):
        self.APP_ID = app_id

    def create_experiment_dir(self, app_id, run_id):
        return experiment_utils._create_experiment_dir(app_id, run_id)

    def get_logdir(self, app_id, run_id):
        return experiment_utils._get_logdir(app_id, run_id)

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
        return experiment_utils._populate_experiment(
            model_name,
            function,
            type,
            hp,
            description,
            app_id,
            direction,
            optimization_key,
        )

    def attach_experiment_xattr(self, ml_id, json_data, op_type):
        return experiment_utils._attach_experiment_xattr(ml_id, json_data, op_type)

    def get_ip_address(self):
        return experiment_utils._get_ip_address()

    def exists(self, hdfs_path, project=None):
        return hopshdfs.exists(hdfs_path, project=project)

    def mkdir(self, hdfs_path, project=None):
        return hopshdfs.mkdir(hdfs_path, project=project)

    def dump(self, data, hdfs_path):
        return hopshdfs.dump(data, hdfs_path)

    def send_request(
        self, method, resource, data=None, headers=None, stream=False, files=None
    ):
        return hopsutil.send_request(
            method, resource, data=data, headers=headers, stream=stream, files=files
        )

    def get_constants(self):
        return hopsconstants

    def open_file(self, hdfs_path, project=None, flags="r", buff_size=0):
        return hopshdfs.open_file(
            hdfs_path, project=project, flags=flags, buff_size=buff_size
        )

    def get_training_dataset_path(
        self, training_dataset, featurestore=None, training_dataset_version=1
    ):
        return featurestore.get_training_dataset_path(
            training_dataset,
            featurestore=None,
            training_dataset_version=training_dataset_version,
        )

    def get_training_dataset_tf_record_schema(
        self, training_dataset, training_dataset_version=1, featurestore=None
    ):
        return featurestore.get_training_dataset_tf_record_schema(
            training_dataset,
            training_dataset_version=training_dataset_version,
            featurestore=featurestore,
        )

    def get_featurestore_metadata(self, featurestore=None, update_cache=False):
        return featurestore.get_featurestore_metadata(
            featurestore=featurestore, update_cache=update_cache
        )

    def init_ml_tracking(self, app_id, run_id):
        tensorboard._register(experiment_utils._get_logdir(app_id, run_id))

    def log_searchspace(self, app_id, run_id, searchspace):
        tensorboard._write_hparams_config(
            experiment_utils._get_logdir(app_id, run_id), searchspace
        )

    def connect_host(self, server_sock, server_host_port, exp_driver):
        if not server_host_port:
            server_sock.bind(("", 0))
            # hostname may not be resolvable but IP address probably will be
            host = self.get_ip_address()
            port = server_sock.getsockname()[1]
            server_host_port = (host, port)
            # register this driver with Hopsworks
            sp = util.find_spark()
            if sp is not None:
                sc = sp.sparkContext
                app_id = str(sc.applicationId)
            else:
                app_id = self.APP_ID
                util.set_app_id(app_id)

            hopscons = self.get_constants()
            method = hopscons.HTTP_CONFIG.HTTP_POST
            resource_url = (
                hopscons.DELIMITERS.SLASH_DELIMITER
                + hopscons.REST_CONFIG.HOPSWORKS_REST_RESOURCE
                + hopscons.DELIMITERS.SLASH_DELIMITER
                + "maggy"
                + hopscons.DELIMITERS.SLASH_DELIMITER
                + "drivers"
            )
            json_contents = {
                "hostIp": host,
                "port": port,
                "appId": app_id,
                "secret": exp_driver._secret,
            }
            json_embeddable = json.dumps(json_contents)
            headers = {
                hopscons.HTTP_CONFIG.HTTP_CONTENT_TYPE: hopscons.HTTP_CONFIG.HTTP_APPLICATION_JSON
            }

            try:
                response = self.send_request(
                    method, resource_url, data=json_embeddable, headers=headers
                )

                if (response.status_code // 100) != 2:
                    print("No connection to Hopsworks for logging.")
                    exp_driver.log("No connection to Hopsworks for logging.")
            except Exception as e:
                print("Connection failed to Hopsworks. No logging.")
                exp_driver.log(e)
                exp_driver.log("Connection failed to Hopsworks. No logging.")
        else:
            server_sock.bind(server_host_port)

        server_sock.listen(10)

        return server_sock, server_host_port

    def get_user(self):
        user = None
        if hopsconstants.ENV_VARIABLES.HOPSWORKS_USER_ENV_VAR in os.environ:
            user = os.environ[hopsconstants.ENV_VARIABLES.HOPSWORKS_USER_ENV_VAR]
        return user

    def project_name(self):
        return hopshdfs.project_name()

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
        """Attaches the experiment outcome as xattr metadata to the app directory."""
        outputs = self.build_summary_json(logdir)

        if outputs:
            self.dump(outputs, logdir + "/.summary.json")

        if best_logdir:
            experiment_json["bestDir"] = best_logdir[len(self.project_path()) :]
        experiment_json["optimizationKey"] = optimization_key
        experiment_json["metric"] = metric
        experiment_json["state"] = state
        experiment_json["duration"] = duration
        exp_ml_id = app_id + "_" + str(run_id)
        self.attach_experiment_xattr(exp_ml_id, experiment_json, "FULL_UPDATE")

    def isdir(self, dir_path, project=None):
        return hopshdfs.isdir(dir_path, project=project)

    def ls(self, dir_path, recursive=False, exclude_nn_addr=None):
        return hopshdfs.ls(
            dir_path, recursive=recursive, exclude_nn_addr=exclude_nn_addr
        )

    def delete(self, path, recursive=False):
        return hopshdfs.delete(path, recursive=recursive)

    def upload_file_output(self, retval, hdfs_exec_logdir):
        return experiment_utils._upload_file_output(retval, hdfs_exec_logdir)

    def project_path(self, project=None, exclude_nn_addr=False):
        return hopshdfs.project_path(project=project, exclude_nn_addr=exclude_nn_addr)

    def str_or_byte(self, str):
        return str.encode()

    def get_executors(self, sc):
        executors = 0
        try:
            executors = int(sc._conf.get("spark.dynamicAllocation.maxExecutors"))
        except TypeError as exc:  # noqa: E722, F841
            pass
        try:
            executors = int(sc._conf.get("spark.executor.instances"))
        except TypeError as exc:  # noqa: E722, F841
            pass
        if executors == 0:
            raise RuntimeError(
                "Failed to find spark.dynamicAllocation.maxExecutors or spark.executor.instances properties, \
                please select jupyter on Spark mode."
            )
        return executors

    def build_summary_json(self, logdir):
        return util.build_summary_json(logdir)

    def convert_return_file_to_arr(self, return_file):
        return experiment_utils._convert_return_file_to_arr(return_file)

    def load(self, hparams_file):
        return hopshdfs.load(hparams_file)

    def connect_hsfs(self, engine="training"):
        return hsfs.connection(engine=engine)
