"""
Maggy specific exceptions.
"""

class EarlyStopException(Exception):
    """Raised by the reporter when a early stop signal is received."""

    def __init__(self, metric):
        super().__init__()
        self.metric = metric

class NotSupportedError(Exception):
    """Raised when we are dealing with a situation that we do not (yet)
    support, e.g., a specific dataset type.
    """

    def __init__(self, category, value, suggestion=""):
        self.message = "({0}: {1}) is not supported by Maggy.{2}".format(
            category, value, suggestion)
        super().__init__(self.message)

class ReturnTypeError(TypeError):
    """User defined training function returns value of wrong type."""

    def __init__(self, optimization_key, return_type):
        self.message = (
            "Training function cannot return value of type: {}. "
            "Return single numeric value or 'dict' containing optimization key"
            " `{}` with numeric value".format(
                type(return_type).__name__, optimization_key))
        super().__init__(self.message)

class MetricTypeError(TypeError):
    """User defined training function returns metric of wrong type."""

    def __init__(self, optimization_key, return_type):
        self.message = (
            "The optimization metric `{}` returned by the training function is"
            " of type: {}. The optimization metric can only be numeric".format(
                optimization_key, type(return_type).__name__))
        super().__init__(self.message)