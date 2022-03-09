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

import tensorflow as tf
from tensorflow import keras
import numpy as np

from maggy.searchspace import Searchspace
from maggy.optimizer import RandomSearch
from maggy import experiment
from maggy.config import HyperparameterOptConfig, TfDistributedConfig

# this allows using the fixture in all tests in this module
pytestmark = pytest.mark.usefixtures("sc")


def test_randomsearch_init():

    sp = Searchspace(argument_param=("DOUBLE", [1, 5]), param2=("integer", [3, 4]))

    rs = RandomSearch(5, sp, [])

    assert rs.num_trials == 5
    assert rs.searchspace == sp


def test_randomsearch_initialize():

    sp = Searchspace(argument_param=("DOUBLE", [1, 5]), param2=("integer", [3, 4]))

    rs = RandomSearch(5, sp, [])

    rs.initialize()

    assert len(rs.trial_buffer) == 5


def test_rs_initialize2():

    sp = Searchspace(argument_param=("DISCRETE", [1, 5]))

    rs = RandomSearch()
    rs.searchspace = sp

    with pytest.raises(NotImplementedError) as excinfo:
        rs.initialize()
    assert "Searchspace needs at least one continuous parameter" in str(excinfo.value)


def test_randomsearch(sc):
    def train(model, train_set, test_set, hparams, reporter):

        if "argument_param" in hparams.keys():
            print(
                "Entered train function with param {}".format(hparams["argument_param"])
            )

        for i in range(5):
            acc = i + random.random()
            reporter.broadcast(metric=acc)
            reporter.log("Metric: {}".format(acc))

            # do something with HP.
            if "argument_param" in hparams.keys():
                time.sleep(hparams["argument_param"])

        return acc

    sp = Searchspace(argument_param=("DOUBLE", [1, 5]))

    config = HyperparameterOptConfig(
        searchspace=sp,
        optimizer="randomsearch",
        direction="max",
        num_trials=5,
        name="test",
        hb_interval=1,
        es_interval=10,
    )

    result = experiment.lagom(train_fn=train, config=config)
    assert type(result) == type({})

    test_dt_tensorflow(sc)


def test_dt_tensorflow(sc):

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.reshape(x_train, (60000, 28, 28, 1))
    x_test = np.reshape(x_test, (10000, 28, 28, 1))

    def training_function(model, train_set, test_set, hparams):
        from tensorflow import keras

        # Define training parameters
        num_epochs = 10
        batch_size = 256
        learning_rate = 0.1

        criterion = keras.losses.SparseCategoricalCrossentropy()
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9, decay=1e-5
        )

        model = model(nlayers=2)

        model.compile(optimizer=optimizer, loss=criterion, metrics=["accuracy"])

        model.fit(
            x_train,
            y_train,
            # batch_size=batch_size,
            # epochs=num_epochs,
        )

        print("Testing")

        loss = model.evaluate(x_test, y_test)

        return loss

    class NeuralNetwork(tf.keras.Model):
        def __init__(self, nlayers):
            super().__init__()
            self.conv1 = keras.layers.Conv2D(28, 2, activation="relu")
            self.flatten = keras.layers.Flatten()
            self.d1 = keras.layers.Dense(32, activation="relu")
            self.d2 = keras.layers.Dense(10, activation="softmax")

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    model = NeuralNetwork

    # define the constructor parameters of your model
    model_parameters = {
        "train_batch_size": 30000,
        "test_batch_size": 5000,
        "nlayers": 2,
    }

    # pass the model parameters in the last
    config = TfDistributedConfig(
        name="tf_test",
        model=model,
        train_set=None,
        test_set=None,
        hparams=model_parameters,
    )

    result = experiment.lagom(train_fn=training_function, config=config)

    assert type(result) == list
