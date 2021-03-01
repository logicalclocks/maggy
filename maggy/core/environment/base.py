import os
import shutil

import maggy.util as util

class BaseEnv:

    def __init__(self):
        self.log_dir = os.path.join(os.getcwd(),'experiment_log')
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.constants = []
        pass

    def set_ml_id(self, app_id = 0, run_id = 0):
        os.environ['ML_ID'] = str(app_id) + '_' + str(run_id)

    def create_experiment_dir(self, app_id, run_id):
        if not os.path.exists(os.path.join(self.log_dir, app_id)):
            os.mkdir(os.path.join(self.log_dir, app_id))

        experiment_path = self.get_logdir(app_id, run_id)
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)

        os.mkdir(experiment_path)

    def get_logdir(self, app_id, run_id):
        return os.path.join(self.log_dir, str(app_id), str(run_id))

    def populate_experiment(self, model_name, function, type, hp, description, app_id, direction, optimization_key):
        pass

    def attach_experiment_xattr(self, exp_ml_id, experiment_json, command):
        pass

    def exists(self, hdfs_path, project=None):
        return os.path.exists(hdfs_path)


    def mkdir(self, hdfs_path, project=None):
        pass

    def isdir(self, dir_path, project=None):
        return os.path.exists(dir_path)

    def ls(self, dir_path, recursive=False, project=None):
        return os.listdir(dir_path)

    def delete(self, path, recursive=False):
        if self.exists(path):
            if os.path.isdir(path):
                os.rmdir(path)
            elif os.path.isfile(path):
                os.remove(path)

    def dump(self, data, hdfs_path):
        head_tail = os.path.split(hdfs_path)
        if not os.path.exists(head_tail[0]):
            os.mkdir(head_tail[0])
        file = self.open_file(hdfs_path, flags='w+')
        file.write(data)

    def get_ip_address(self):
        sc = util.find_spark().sparkContext
        return sc._conf.get("spark.driver.host")

    def get_constants(self):
        pass

    def open_file(self, hdfs_path, project=None, flags='rw', buff_size=0):
        return open(hdfs_path, mode=flags)

    def get_training_dataset_path(self, training_dataset, featurestore=None, training_dataset_version=1):
        pass

    def get_training_dataset_tf_record_schema(self, training_dataset, training_dataset_version=1, featurestore=None):
        pass

    def get_featurestore_metadata(self, featurestore=None, update_cache=False):
        pass

    def init_ml_tracking(self, app_id, run_id):
        pass

    def log_searchspace(self, app_id, run_id, searchspace):
        pass

    def connect_host(self,server_sock, server_host_port, exp_driver):
        if not server_host_port:
            server_sock.bind(("", 0))
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

    def finalize_experiment(self,
            experiment_json,
            metric,
            app_id,
            run_id,
            state,
            duration,
            logdir,
            best_logdir,
            optimization_key
                            ):
        pass

    def str_or_byte(self, str):
        return str

    def get_executors(self, sc):
        try:
            if sc._conf.get("spark.dynamicAllocation.enabled") == "true":
                maxExecutors = int(sc._conf.get("spark.dynamicAllocation.maxExecutors"))
            else:
                maxExecutors = int(sc._conf.get("spark.executor.instances"))

            return maxExecutors
        except:  # noqa: E722
            raise RuntimeError(
                "Failed to find some of the spark.databricks properties."
            )

    def build_summary_json(self):
        pass

    def connect_hsfs(self):
        pass

    def convert_return_file_to_arr(self):
        pass

    def upload_file_output(self, retval, hdfs_exec_logdir):
        pass