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

import pytest
import time
import random

from maggy import Trial


def test_trial_init():

    trial = Trial({"param1": 5, "param2": "ada"})

    exp = {"param1": 5, "param2": "ada"}

    assert trial.params == exp
    assert trial.status == Trial.PENDING
    assert trial.trial_id == "3d1cc9fdb1d4d001"


def test_trial_serialization():

    trial = Trial({"param1": 5, "param2": "ada"})

    exp = {"param1": 5, "param2": "ada"}

    json_str = trial.to_json()

    new_trial = Trial.from_json(json_str)

    assert isinstance(new_trial, Trial)
    assert new_trial.params == exp
    assert new_trial.status == Trial.PENDING
    assert new_trial.trial_id == "3d1cc9fdb1d4d001"
