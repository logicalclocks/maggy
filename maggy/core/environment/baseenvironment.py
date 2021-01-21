


from maggy.core.environment.abstractenvironment import AbstractEnvironment
import os
import socket

class BaseEnvironment(AbstractEnvironment):

    def __init__(self, *args):
        self.log_dir = "/databricks/driver/logs"
        self.constants = []
        pass

    def set_ml_id(self, app_id = 0, run_id = 0):
        os.environ['ML_ID'] = app_id + '_' + str(run_id)

    def create_experiment_dir(self, app_id, run_id):
        pass

    def get_logdir(self, app_id, run_id):
        return self.log_dir

    def populate_experiment(self, model_name, function, type, hp, description, app_id, direction, optimization_key):
        pass

    def attach_experiment_xattr(self, exp_ml_id, experiment_json, command):
        pass

    def exists(self, hdfs_path, project=None):
        pass

    def mkdir(self, hdfs_path, project=None):
        pass

    def dump(self, data, hdfs_path):
        pass

    def get_ip_address(self):
        try:
            _, _, _, _, addr = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM)[0]
            return addr[0]
        except:
            return socket.gethostbyname(socket.getfqdn())

    def get_constants(self):
        pass

    def open_file(self, hdfs_path, project=None, flags='rw', buff_size=0):
        return open(hdfs_path,mode=flags)

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
            host = self.get_ip_address()
            port = server_sock.getsockname()[1]
            server_host_port = (host, port)
            print("(not server_host_post) = True, server_host_port = {} ".format(server_host_port))
            server_sock.bind(server_host_port)

        else:
            print("server_host_post = True, server_host_port = {} ".format(server_host_port))
            server_sock.bind(server_host_port)

        server_sock.listen(10)

        return server_sock, server_host_port