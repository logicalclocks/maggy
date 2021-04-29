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

from typing import Union, Callable

from maggy.experiment_config import LagomConfig

import tensorflow as tf


class TfDistributedConfig(LagomConfig):
    def __init__(
        self,
        model: tf.keras.Model,
        train_set: Union[str, tf.data.Dataset],
        test_set: Union[str, tf.data.Dataset],
        process_data: Callable = None,
        mixed_precision: bool = False,
        name: str = "tfDist",
        hb_interval: int = 1,
        description: str = "",
        hparams: dict = None,
    ):

        """Initializes Tensorflow distributed training parameters.

        :param model: A tf.keras.Model superclass or list of them.
            Note that this has to be the class itself, not an instance.
        :param train_set: The training set for the training function. If you want to load the set
            inside the training function, this can be disregarded.
        :param test_set: The test set for the training function. If you want to load the set
            inside the training function, this can be disregarded.
        :param process_data: The function for processing the data
        :param hparams: model parameters that should be used during model initialization. Primarily
            used to give an interface for hp optimization.
        :param name: Experiment name.
        :param hb_interval: Heartbeat interval with which the server is polling.
        :param description: A description of the experiment.
        """
        super().__init__(name, description, hb_interval)
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.process_data = process_data
        self.mixed_precision = mixed_precision
        self.hparams = hparams if hparams else {}
