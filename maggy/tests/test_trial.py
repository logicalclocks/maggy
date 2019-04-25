import pytest
import time
import random

from maggy import Trial

def test_trial_init():

    trial = Trial({'param1': 5, 'param2': 'ada'})

    exp = {'param1': 5, 'param2': 'ada'}

    assert trial.params == exp
    assert trial.status == Trial.PENDING
    assert trial.trial_id == '3d1cc9fdb1d4d001'

def test_trial_serialization():

    trial = Trial({'param1': 5, 'param2': 'ada'})

    exp = {'param1': 5, 'param2': 'ada'}

    json_str = trial.to_json()

    new_trial = Trial.from_json(json_str)

    assert isinstance(new_trial, Trial)
    assert new_trial.params == exp
    assert new_trial.status == Trial.PENDING
    assert new_trial.trial_id == '3d1cc9fdb1d4d001'