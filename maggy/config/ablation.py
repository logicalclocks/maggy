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

from typing import Union, List

from maggy.ablation.ablationstudy import AblationStudy
from maggy.ablation.ablator import AbstractAblator
from maggy.config import LagomConfig
import tensorflow as tf
from maggy.core import config as mc


class AblationConfig(LagomConfig):
    """Config class for ablation study experiments."""

    def __init__(
        self,
        ablation_study: AblationStudy,
        ablator: Union[str, AbstractAblator] = "loco",
        direction: str = "max",
        name: str = "ablationStudy",
        description: str = "",
        hb_interval: int = 1,
        model: tf.keras.Model = None,
        dataset: List[Union[str, tf.data.Dataset]] = None,
    ):
        """Initializes ablation study experiment parameters.

        :param ablation_study: Ablation study object that defines the entry point into the
            experiment.
        :param ablator: An instance of `AbstractAblator` or a supported ablator name that controls
            the manner in which parts of the model are ablated.
        :param direction: Optimization direction to evaluate the experiments.
        :param name: Experiment name.
        :param description: A description of the experiment.
        :param hb_interval: Heartbeat interval with which the server is polling.
        :param model: The class of the model to be used in the training function.
        :param dataset: A List of strings containing the dataset path or list of tf.data.Dataset.
        These datasets represent the ones you are going to use in the training function.
        For example, if you have 2 datasets for training and testing, pass an array with [train_set, test_set] and
        extract them in the training function. If you want to load the set inside the training function, this can be
        disregarded.
        """
        super().__init__(name, description, hb_interval)
        mc.initialize()
        if not mc.is_spark_available():
            raise NotImplementedError("Ablation Study can run only on a Spark kernel.")
        self.ablator = ablator
        self.ablation_study = ablation_study
        self.direction = direction
        self.model = model
        self.dataset = dataset
