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
