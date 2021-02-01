from maggy.core.environment import abstractenvironment, baseenvironment
import os

AbstractEnvironment = abstractenvironment.AbstractEnvironment

if "REST_ENDPOINT" in os.environ:
    print("You are running maggy on Hopsworks.")

    import hopsenvironment
    HopsEnvironment = hopsenvironment.HopsEnvironment

    __all__ = ["AbstractEnvironment", "HopsEnvironment"]

#todo: condition for databricks
elif False:
    print("You are running maggy on Databricks.")

    DatabricksEnvironment = databricksenvironment.DatabricksEnvironment

    __all__ = ["AbstractEnvironment", "DatabricksEnvironment"]
else:
    print("You are running maggy spark-only configuration.")
    BaseEnvironment = baseenvironment.BaseEnvironment

    __all__ = ["AbstractEnvironment", "BaseEnvironment"]


