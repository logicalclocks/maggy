


from maggy.core.environment.abstractenvironment import AbstractEnvironment
import os
import socket
import maggy.util as util
import pyspark
import uuid
import tensorflow as tf

class BaseEnvironment(AbstractEnvironment):
    """
    Environment implemented for maggy usage on Databricks.
    """

    def __init__(self, *args):
        self.log_dir = "/dbfs/maggy_log/"
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
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
        return os.path.exists(hdfs_path)


    def mkdir(self, hdfs_path, project=None):
        pass

    def isdir(self, dir_path):
        return os.path.exists(dir_path)

    def ls(self, dir_path):
        _ , dirnames, filenames = next(os.walk(dir_path))

        return dirnames + filenames

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
        sc = util._find_spark().sparkContext
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
        return "/dbfs/"

    def get_user(self):
        # TODO retrieve user info from databricks
        return ""

    def project_name(self):
        # TODO retrieve project_name from databricks
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
            if sc._conf.get("spark.databricks.clusterUsageTags.clusterScalingType") == "autoscaling":
                maxExecutors = int(sc._conf.get("spark.databricks.clusterUsageTags.clusterMaxWorkers"))
            else:
                maxExecutors = int(sc._conf.get("spark.databricks.clusterUsageTags.clusterWorkers"))

            return maxExecutors
        except:  # noqa: E722
            raise RuntimeError(
                "Failed to find some of the spark.databricks properties."
            )

    def _build_summary_json(self, logdir):
        pass

    def _convert_return_file_to_arr(self, return_file):
        pass

    def connect_hsfs(self, engine="training"):
        pass

    def create_tf_dataset(self, num_epochs, batch_size,
                          training_dataset_version, training_dataset_name,
                          ablated_feature, label_name, training_dataset_path):


        spark_df = pyspark.read.csv(training_dataset_path, header="true", inferSchema="true")


        feature_names = spark_df.schema.names

        if ablated_feature is not None:
            feature_names.remove(ablated_feature)

        name_uuid = str(uuid.uuid4())
        path = '/ml/'+ training_dataset_name + '/df-{}.tfrecord'.format(name_uuid)
        spark_df.write.format("tfrecords").mode("overwrite").save(path)

        return self.load_tf_dataset(path)

    def load_tf_dataset(self,path):
        filenames = [("/dbfs" + path + "/" + name) for name in os.listdir("/dbfs" + path) if name.startswith("part")]
        dataset = tf.data.TFRecordDataset(filenames)
        return dataset

    def connect_hsfs(self,engine='engine'):
        pass