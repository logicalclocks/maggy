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

from maggy import Searchspace


def test_searchspace_init():

    sp = Searchspace(argument_param=("DOUBLE", [1, 5]), param2=("integer", [3, 4]))

    exp_get = [1, 5]

    assert sp.get("argument_param") == exp_get
    assert sp.argument_param == exp_get  # pylint: disable=no-member


def test_searchspace_add():

    sp = Searchspace(argument_param=("DOUBLE", [1, 5]))

    with pytest.raises(ValueError) as excinfo:
        sp.add("argument_param", ("DOUBLE", [1, 5]))
    assert "Hyperparameter name is reserved" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # add tuple with too many elements
        sp.add("param", ("DOUBLE", [1, 5], "too many"))
    assert "Hyperparameter tuple has to be of length two" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # add unknown type
        sp.add("param", ("FLOAT", [1, 5]))
    assert "Hyperparameter type is not of type " in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # add empty region list
        sp.add("param", ("DOUBLE", []))
    assert "Hyperparameter feasible region list" in str(excinfo.value)

    with pytest.raises(AssertionError) as excinfo:
        # add incompatible type and feasible region
        sp.add("param", ("DOUBLE", [1, 5, 5]))
        sp.add("param2", ("INTEGER", [1, 5, 5]))
    assert "For DOUBLE or " in str(excinfo.value)

    with pytest.raises(AssertionError) as excinfo:
        # lower bound higher than upper bound
        sp.add("param", ("DOUBLE", [5, 1]))
        sp.add("param2", ("INTEGER", [4, 1]))
    assert "Lower bound " in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # Non integer boundaries for integer type parameter
        sp.add("param2", ("INTEGER", [1.5, 5]))
    assert "type INTEGER need to be integer:" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # Non numeric interval boundaries
        sp.add("param2", ("DOUBLE", ["lower", 5]))
    assert "type DOUBLE need to be integer or float:" in str(excinfo.value)
