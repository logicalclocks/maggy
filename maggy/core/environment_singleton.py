
def _is_hops_available():
    try:
        import hops
    except ModuleNotFoundError:
        return False
    except ImportError:
        return False

    return True

class EnvironmentSingleton:
    __instance = None
    def __new__(cls, *args):
        if cls.__instance is None:
            # check hopsworks availability
            if _is_hops_available():

                from maggy.core.environment import HopsEnvironment
                cls.__instance = HopsEnvironment(cls, *args)
                print("Hopsworks APIs are available.")
            else:
                from maggy.core.environment import BaseEnvironment
                cls.__instance = BaseEnvironment(cls, *args)
                print("Hopsoworks APIs are not available, using spark-only configuration.")

            if cls.__instance is None:
                raise AttributeError("Environment is None.")

        return cls.__instance


