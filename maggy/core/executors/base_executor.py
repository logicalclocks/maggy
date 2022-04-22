#
#   Copyright 2021 Logical Clocks AB
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

from typing import Callable, Any


def base_executor_fn(
    train_fn: Callable,
) -> Callable:
    """Wraps the user supplied training function in order to be passed to the Spark Executors.

    :param train_fn: Original training function.
    :param config: Experiment config.

    :returns: Patched function to execute on the Spark executors.
    """

    def wrapper_function(_: Any) -> None:
        """Patched function from tf_dist_executor_fn factory.

        :param _: Necessary catch for the iterator given by Spark to the
        function upon foreach calls. Can safely be disregarded.
        """

        retval = train_fn()
        return retval

    return wrapper_function
