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

from typing import Union, Callable, List, Optional

from maggy.config import LagomConfig

import tensorflow as tf


class TfDistributedConfig(LagomConfig):
    def __init__(
        self,
        model: tf.keras.Model = None,
        dataset: List[Optional[Union[str, tf.data.Dataset]]] = None,
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
        :param dataset: A List of strings containing the dataset path or list of tf.data.Dataset.
        these datasets represent the ones you are going to use in the training function. For example,
        if you have 2 datasets for training and testing, pass an array with [train_set, test_set] and extract them in
        the training function. If you want to load the set inside the training function, this can be disregarded.
        :param process_data: The function for processing the data.
        :param hparams: model parameters that should be used during model initialization. Primarily
            used to give an interface for hp optimization.
        :param name: Experiment name.
        :param hb_interval: Heartbeat interval with which the server is polling.
        :param description: A description of the experiment.
        """
        super().__init__(name, description, hb_interval)
        self.model = model
        self.dataset = dataset
        self.process_data = process_data
        self.mixed_precision = mixed_precision
        self.hparams = hparams if hparams else {}
