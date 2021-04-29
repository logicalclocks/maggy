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


def get_wrapped_model(model, strategy):
    """Build a wrap class for the user defined tensorflow model.

    :param model: The class of the user defined tensorflow model.
    :param strategy: A class of the strategy to be used for the training.

    :returns: The TensorflowModelWrapper class.
    """

    class TensorflowModelWrapper(model):
        """A wrap for tensorflow model, the __init__() and compile() functions are overridden in order to launch
        train the model in a distributed fashion.
        """

        def __init__(self, *args, **kwargs):
            self.__strategy = strategy
            with self.__strategy.scope():
                try:
                    super().__init__(*args, **kwargs)
                except TypeError as e:
                    raise TypeError(
                        "The parameters passed to TensorflowConfig (model_parameters) "
                        "do not corresponds to the parameters defined in your model "
                        "constructor."
                    ) from e

        def compile(
            self,
            optimizer="rmsprop",
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            **kwargs
        ):
            """This method wraps the compile function from tensorflow,
            adding the distributed strategy scope before the compile() call of tensorflow.
            """
            with self.__strategy.scope():
                super().compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=metrics,
                    loss_weights=loss_weights,
                    weighted_metrics=weighted_metrics,
                    run_eagerly=run_eagerly,
                    steps_per_execution=steps_per_execution,
                    **kwargs
                )

    return TensorflowModelWrapper
