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

from __future__ import annotations

from maggy.config import LagomConfig
from maggy.core import config as mc


class BaseConfig(LagomConfig):
    def __init__(
        self,
        name: str = "base",
        hb_interval: int = 1,
        description: str = "",
    ):

        """Initializes Base configuration.

        :param name: Experiment name.
        :param hb_interval: Heartbeat interval with which the server is polling.
        :param description: A description of the experiment.
        """
        super().__init__(name, description, hb_interval)
        mc.initialize()
