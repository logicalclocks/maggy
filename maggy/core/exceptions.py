#
#   Copyright 2020 Logical Clocks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

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
            category, value, suggestion
        )
        super().__init__(self.message)


class ReturnTypeError(TypeError):
    """User defined training function returns value of wrong type."""

    def __init__(self, optimization_key, return_type):
        self.message = (
            "Training function cannot return value of type: {}. "
            "Return single numeric value or 'dict' containing optimization key"
            " `{}` with numeric value".format(
                type(return_type).__name__, optimization_key
            )
        )
        super().__init__(self.message)


class MetricTypeError(TypeError):
    """User defined training function returns metric of wrong type."""

    def __init__(self, optimization_key, return_type):
        self.message = (
            "The optimization metric `{}` returned by the training function is"
            " of type: {}. The optimization metric can only be numeric".format(
                optimization_key, type(return_type).__name__
            )
        )
        super().__init__(self.message)


class BadArgumentsError(Exception):
    """Raised when a function or method has been called with incompatible arguments.
    This can be used by developers to prevent bad usage of their functions
    or classes by other developers.
    """

    def __init__(self, callable, suggestion=""):
        self.message = "{0} was called using incompatible arguments. {1}".format(
            callable, suggestion
        )
        super().__init__(self.message)
