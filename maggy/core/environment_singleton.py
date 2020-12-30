
from maggy.core import environment

class EnvironmentSingleton:
    __instance = None
    def __new__(cls, *args):
        if cls.__instance is None:
            # check hopsworks availability
            try:
                from hops import util as hopsutil
                is_hops = True
            except ImportError:
                is_hops = False

            if is_hops:
                cls.__instance = environment.HopsEnvironment(cls, *args)
                print("Hopsworks APIs are available.")
            else:
                cls.__instance = environment.BaseEnvironment(cls, *args)
                print("Hopsoworks APIs are not available, using spark-only configuration")

        return cls.__instance