


from hops import util as hopsutil
from hops.experiment_impl.util import experiment_utils
from hops import hdfs as hopshdfs
from hops import constants
from maggy import tensorboard
from maggy import util

import json

from maggy.core.environment.abstractenvironment import AbstractEnvironment



class HopsEnvironment(AbstractEnvironment):


    def __init__(self, *args):
        self.constants = constants


    def set_ml_id(self, app_id, run_id):
        return experiment_utils._set_ml_id(app_id,run_id)

    def create_experiment_dir(self, app_id, run_id):
        return experiment_utils._create_experiment_dir(app_id,run_id)

    def get_logdir(self, app_id, run_id):
        return experiment_utils._get_logdir(app_id,run_id)

    def populate_experiment(self, model_name, function, type, hp, description, app_id, direction, optimization_key):
        return experiment_utils._populate_experiment(model_name, function, type, hp, description, app_id, direction, optimization_key)

    def attach_experiment_xattr(self, ml_id, json_data, op_type):
        return experiment_utils._attach_experiment_xattr(ml_id, json_data, op_type)

    def get_ip_address(self):
        return experiment_utils._get_ip_address()

    def exists(self, hdfs_path, project=None):
        return hopshdfs.exists(hdfs_path, project=project)

    def mkdir(self, hdfs_path, project=None):
        return hopshdfs.mkdir(hdfs_path, project=project)

    def dump(self, data, hdfs_path):
        return hopshdfs.dump(data,hdfs_path)

    def send_request(self, method, resource, data=None, headers=None, stream=False, files=None):
        return hopsutil.send_request(method, resource, data=data, headers=headers, stream=stream, files=files)

    def get_constants(self):
        return self.constants

    def open_file(self, hdfs_path, project=None, flags='rw', buff_size=0):
        return hopshdfs.open_file(hdfs_path, project=project, flags=flags, buff_size=buff_size)


    def get_training_dataset_path(self, training_dataset, featurestore=None, training_dataset_version=1):
        return featurestore.get_training_dataset_path(training_dataset, featurestore=None,
                                                      training_dataset_version=training_dataset_version)


    def get_training_dataset_tf_record_schema(self, training_dataset, training_dataset_version=1, featurestore=None):
        return featurestore.get_training_dataset_tf_record_schema(training_dataset,
                                                                  training_dataset_version=training_dataset_version,
                                                                  featurestore=featurestore)


    def get_featurestore_metadata(self, featurestore=None, update_cache=False):
        return featurestore.get_featurestore_metadata(featurestore=featurestore, update_cache=update_cache)

    def init_ml_tracking(self, app_id, run_id):
        tensorboard._register(experiment_utils._get_logdir(app_id, run_id))

    def log_searchspace(self, app_id, run_id, searchspace):
        tensorboard._write_hparams_config(self.get_logdir(app_id, run_id), searchspace)

    def connect_host(self,server_sock,server_host_port, exp_driver):
        if not server_host_port:
            server_sock.bind(("", 0))
            # hostname may not be resolvable but IP address probably will be
            host = self.get_ip_address()
            port = server_sock.getsockname()[1]
            server_host_port = (host, port)
            # register this driver with Hopsworks
            sc = util._find_spark().sparkContext
            app_id = str(sc.applicationId)

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
                    exp_driver._log("No connection to Hopsworks for logging.")
            except Exception as e:
                print("Connection failed to Hopsworks. No logging.")
                exp_driver._log(e)
                exp_driver._log("Connection failed to Hopsworks. No logging.")
        else:
            server_sock.bind(server_host_port)

        server_sock.listen(10)

        return server_sock, server_host_port