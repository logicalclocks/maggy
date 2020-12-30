



from abc import ABC, abstractmethod


class AbstractEnvironment(ABC,):
    def __init__(self, *args):
        pass

    def _set_ml_id(self, app_id, run_id):
        pass

    def _create_experiment_dir(self,app_id,run_id):
        pass

    def _get_logdir(self,app_id,run_id):
        pass

    def _populate_experiment(self,  model_name, function, type, hp, description, app_id, direction, optimization_key):
        pass

    def _attach_experiment_xattr(self, exp_ml_id, experiment_json, command):
        pass

    def exists(self, hdfs_path, project=None):
        pass

    def mkdir(self, hdfs_path, project=None):
        pass

    def dump(self, data, hdfs_path):
        pass

    def _get_ip_address(self):
        pass

    def get_constants(self):
        pass

    def open_file(self, hdfs_path, project=None, flags='rw', buff_size=0):
        pass