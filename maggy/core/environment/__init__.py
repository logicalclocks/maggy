from maggy.core.environment import abstractenvironment, baseenvironment

try:
    import hopsenvironment
    HopsEnvironment = hopsenvironment.HopsEnvironment
    AbstractEnvironment = abstractenvironment.AbstractEnvironment
    BaseEnvironment = baseenvironment.BaseEnvironment

    __all__ = ["AbstractEnvironment", "HopsEnvironment", "BaseEnvironment"]
except ModuleNotFoundError:
    AbstractEnvironment = abstractenvironment.AbstractEnvironment
    BaseEnvironment = baseenvironment.BaseEnvironment

    __all__ = ["AbstractEnvironment", "BaseEnvironment"]
except ImportError:
    AbstractEnvironment = abstractenvironment.AbstractEnvironment
    BaseEnvironment = baseenvironment.BaseEnvironment

    __all__ = ["AbstractEnvironment", "BaseEnvironment"]


