


from hops import util as hopsutil
from hops.experiment_impl.util import experiment_utils
from hops import hdfs as hopshdfs
from hops import constants

from maggy.core.environment.abstractenvironment import AbstractEnvironment


class HopsEnvironment(AbstractEnvironment):


    def __init__(self, *args):
        self.constants = constants


    def _set_ml_id(self, app_id, run_id):
        return experiment_utils._set_ml_id(app_id,run_id)

    def _create_experiment_dir(self,app_id,run_id):
        return experiment_utils._create_experiment_dir(app_id,run_id)

    def _get_logdir(self,app_id,run_id):
        return experiment_utils._get_logdir(app_id,run_id)

    def _populate_experiment(self, model_name, function, type, hp, description, app_id, direction, optimization_key):
        return experiment_utils._populate_experiment(model_name, function, type, hp, description, app_id, direction, optimization_key)

    def _attach_experiment_xattr(self, ml_id, json_data, op_type):
        return experiment_utils._attach_experiment_xattr((ml_id, json_data, op_type))

    def _get_ip_address(self):
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
