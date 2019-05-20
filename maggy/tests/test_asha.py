import pytest
import time
import random

from maggy.searchspace import Searchspace
from maggy.optimizer import Asha
from maggy import experiment

# this allows using the fixture in all tests in this module
pytestmark = pytest.mark.usefixtures("sc")

def test_asha_init():

    sp = Searchspace(promote=('DISCRETE', [5]), resource=('INTEGER', [1, 25]))

    asha = Asha(25, sp, [])

    asha.initialize()

    assert asha.resource_min == 1
    assert asha.resource_max == 25
    assert asha.promote == 5