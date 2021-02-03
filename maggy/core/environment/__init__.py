from maggy.core.environment import abstractenvironment, baseenvironment
import os
import json
from maggy import util

AbstractEnvironment = abstractenvironment.AbstractEnvironment

if "REST_ENDPOINT" in os.environ:
    print("You are running maggy on Hopsworks.")

    from maggy.core.environment import hopsenvironment
    HopsEnvironment = hopsenvironment.HopsEnvironment

    __all__ = ["AbstractEnvironment", "HopsEnvironment"]

else:
    print("You are running maggy spark-only configuration.")
    BaseEnvironment = baseenvironment.BaseEnvironment

    __all__ = ["AbstractEnvironment", "BaseEnvironment"]


