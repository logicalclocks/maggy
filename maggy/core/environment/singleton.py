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


class EnvSing(object):

    obj = None

    def __new__(cls, *args, **kwargs):
        if EnvSing.obj is not None:
            raise Exception("A Test Singleton instance already exists")

        # check hopsworks availability
        if "REST_ENDPOINT" in os.environ:
            print("Detected Environment: Hopsworks.")

            from maggy.core.environment import hopsworks

            EnvSing.obj = hopsworks.HopsworksEnv()

        elif os.environ.get("DATABRICKS_ROOT_CONDA_ENV") == "databricks-ml":
            print("Detected Environment: Databricks.")

            from maggy.core.environment import databricks

            EnvSing.obj = databricks.DatabricksEnv()

        else:
            print("Detected Environment: base.")

            from maggy.core.environment import base

            EnvSing.obj = base.BaseEnv()

        if EnvSing.obj is None:
            raise NotImplementedError(
                "environment_instance is None, environment not initialised."
            )

    @staticmethod
    def get_instance():
        """
        return an instance of the environment to be used by maggy within a session.
        """
        if EnvSing.obj is None:
            EnvSing()
        return EnvSing.obj
