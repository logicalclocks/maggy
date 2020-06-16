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


# TODO: Discuss naming, this is a draft
class Controller(ABC):
    @abstractmethod
    def initialize(self):
        # TODO: potentially add argument "controller_kwargs" to lagom to allow user
        # to pass through arguments to this method
        pass

    @abstractmethod
    def get_next_trial(self, trial=None):
        # TODO: decide on name, also exp driver has 'get_trial', shorter maybe 'next_trial'
        pass

    @abstractmethod
    def finalize(self, trials):
        # TODO: allow developer to return random dict, e.g. maybe move logic
        # to generate result dict here (but then it will be duplicated)
        # or provide base implementation, that user can overwrite
        pass

    def name(self):
        return str(self.__class__.__name__)
