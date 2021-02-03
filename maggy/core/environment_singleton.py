import os
import json
# in case importing in %%local
try:
    from pyspark.sql import SparkSession
except:
    pass

def environment_singleton():
        global environmentInstance
        if not 'environmentInstance' in globals():
            # check hopsworks availability
            if "REST_ENDPOINT" in os.environ:
                from maggy.core.environment import HopsEnvironment
                environmentInstance = HopsEnvironment()

            else:
                from maggy.core.environment import BaseEnvironment
                environmentInstance = BaseEnvironment()

        if not 'environmentInstance' in globals():
            raise ValueError("environmentInstance is not defined")

        if environmentInstance is None :
            raise AttributeError("environmentInstance is None")

        return environmentInstance
