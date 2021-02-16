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

import os

"""
return an instance of the environment to be used by maggy within a session.
"""


def environment_singleton():
    global environment_instance
    if not "environmentInstance" in globals():
        # check hopsworks availability
        if "REST_ENDPOINT" in os.environ:
            from maggy.core.environment import HopsEnvironment

            environment_instance = HopsEnvironment()

        else:
            from maggy.core.environment import BaseEnvironment

            environment_instance = BaseEnvironment()

    if not "environment_instance" in globals():
        raise ValueError("environment_instance is not defined")

    if environment_instance is None:
        raise AttributeError("environment_instance is None")

    return environment_instance
