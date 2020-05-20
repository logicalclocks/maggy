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

from abc import ABC, abstractmethod


class AbstractEarlyStop(ABC):
    """An Abstract class to implement custom early stopping criteria."""

    @staticmethod
    @abstractmethod
    def earlystop_check(to_check, finalized_trials, direction):
        """A abstract static method that needs to be implemented with a custom
        early stopping criterium.

        The function is called internally in the user specified interval
        with three arguments. It is necessary to add these to the function
        definition.

        :param to_check: A dictionary of currently running
        trials, where the key is the `trial_id` and values are Trial objects.
        :type to_check: dictionary
        :param finalized_trials: A list of finalized Trial objects.
        :type finalized_trials: list
        :param direction: A string describing the search objective, i.e. 'min'
        or 'max'.
        :type direction: str
        """
        pass
