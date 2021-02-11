import os

from maggy.core.environment import abstractenvironment, baseenvironment

AbstractEnvironment = abstractenvironment.AbstractEnvironment
# check which environment to use
if "REST_ENDPOINT" in os.environ:
    print("You are running maggy on Hopsworks.")

    from maggy.core.environment import hopsenvironment

    HopsEnvironment = hopsenvironment.HopsEnvironment

    __all__ = ["AbstractEnvironment", "HopsEnvironment"]

else:
    print("You are running maggy without hopsworks.")
    BaseEnvironment = baseenvironment.BaseEnvironment

    __all__ = ["AbstractEnvironment", "BaseEnvironment"]
