import pytest
import time
import random

from maggy.searchspace import Searchspace
from maggy.optimizer import RandomSearch
from maggy import experiment

# this allows using the fixture in all tests in this module
pytestmark = pytest.mark.usefixtures("sc")

def test_randomsearch_init():

    sp = Searchspace(argument_param=('DOUBLE', [1, 5]), param2=('integer', [3, 4]))

    rs = RandomSearch(5, sp)

    assert rs.num_trials == 5
    assert rs.searchspace == sp

def test_randomsearch_initialize():

    sp = Searchspace(argument_param=('DOUBLE', [1, 5]), param2=('integer', [3, 4]))

    rs = RandomSearch(5, sp)

    rs.initialize()

    assert len(rs.trial_buffer) == 5

def test_rs_initialize2():

    sp = Searchspace(argument_param=('DISCRETE', [1, 5]))

    rs = RandomSearch(5, sp)

    with pytest.raises(NotImplementedError) as excinfo:
        rs.initialize()
    assert "Searchspace needs at least one continuous parameter" in str(excinfo.value)

def test_randomsearch(sc):

    sp = Searchspace(argument_param=('DOUBLE', [1, 5]))

    def train(argument_param, reporter):

        print("Entered train function with param {}".format(argument_param))

        for i in range(5):
            acc = i+random.random()
            reporter.broadcast(metric=acc)

            time.sleep(argument_param)

        return acc

    result = experiment.launch(train, sp, optimizer='randomsearch', direction='max', num_trials=25, name='test', hb_interval=1, es_interval=10)
    assert type(result) == type({})