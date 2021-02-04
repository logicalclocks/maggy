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

"""
Module to encapsulate functionality related to writing to the tensorboard
log dir and programmatically structure the outputs.
"""

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

_tensorboard_dir = None


def _register(trial_dir):
    global _tensorboard_dir
    _tensorboard_dir = trial_dir


def logdir():
    """Returns the path to the tensorboard log directory.

    Instead of hardcoding a log dir path in a training function, users should
    make use of this function call, which will programmatically create a folder
    structure for tensorboard to visualize the machine learning experiment.

    :return: Path of the log directory in HOPSFS
    :rtype: str
    """
    global _tensorboard_dir
    return _tensorboard_dir


def _create_hparams_config(searchspace):
    hparams = []

    for key, val in searchspace.names().items():
        if val == "DOUBLE":
            hparams.append(
                hp.HParam(
                    key,
                    hp.RealInterval(
                        float(searchspace.get(key)[0]), float(searchspace.get(key)[1])
                    ),
                )
            )
        elif val == "INTEGER":
            hparams.append(
                hp.HParam(
                    key,
                    hp.IntInterval(searchspace.get(key)[0], searchspace.get(key)[1]),
                )
            )
        elif val == "DISCRETE":
            hparams.append(hp.HParam(key, hp.Discrete(searchspace.get(key))))
        elif val == "CATEGORICAL":
            hparams.append(hp.HParam(key, hp.Discrete(searchspace.get(key))))

    return hparams


def _write_hparams_config(log_dir, searchspace):
    HPARAMS = _create_hparams_config(searchspace)
    METRICS = [
        hp.Metric(
            "epoch_accuracy", group="validation", display_name="accuracy (val.)",
        ),
        hp.Metric("epoch_loss", group="validation", display_name="loss (val.)",),
        hp.Metric("epoch_accuracy", group="train", display_name="accuracy (train)",),
        hp.Metric("epoch_loss", group="train", display_name="loss (train)",),
    ]

    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(hparams=HPARAMS, metrics=METRICS)


def _write_hparams(hparams, trial_id):
    global _tensorboard_dir
    with tf.summary.create_file_writer(_tensorboard_dir).as_default():
        hp.hparams(hparams, trial_id)
