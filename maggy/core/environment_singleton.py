import os

"""
return an instance of the environment to be used by maggy within a session.
"""


def environment_singleton():
    global environmentInstance
    if not "environmentInstance" in globals():
        # check hopsworks availability
        if "REST_ENDPOINT" in os.environ:
            from maggy.core.environment import HopsEnvironment

            environmentInstance = HopsEnvironment()

        else:
            from maggy.core.environment import BaseEnvironment

            environmentInstance = BaseEnvironment()

    if not "environmentInstance" in globals():
        raise ValueError("environmentInstance is not defined")

    if environmentInstance is None:
        raise AttributeError("environmentInstance is None")

    return environmentInstance
