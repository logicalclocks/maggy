import os

"""
return an instance of the environment to be used by maggy within a session.
"""


def environment_singleton():
    global environment_instance
    if not "environmentInstance" in globals():
        # check hopsworks availability
        if "REST_ENDPOINT" in os.environ:
            from maggy.core.environment import HopsEnvironment

            environment_instance = HopsEnvironment()

        else:
            from maggy.core.environment import BaseEnvironment

            environment_instance = BaseEnvironment()

    if not "environment_instance" in globals():
        raise ValueError("environment_instance is not defined")

    if environment_instance is None:
        raise AttributeError("environment_instance is None")

    return environment_instance
