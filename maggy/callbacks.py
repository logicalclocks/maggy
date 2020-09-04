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

import tensorflow as tf


class KerasBatchEnd(tf.keras.callbacks.Callback):
    """A Keras callback reporting a specified `metric` at the end of the batch
    to the maggy experiment driver.

    `loss` is always available as a metric, and optionally `acc` (if accuracy
    monitoring is enabled, that is, accuracy is added to keras model metrics).
    Validation metrics are not available for the BatchEnd callback. Validation
    after every batch would be too expensive.
    Default is training loss (`loss`).

    Example usage:

    >>> from maggy.callbacks import KerasBatchEnd
    >>> callbacks = [KerasBatchEnd(reporter, metric='acc')]
    """

    def __init__(self, reporter, metric="loss"):
        super().__init__()
        self.metric_name = metric
        self.reporter = reporter

    def on_batch_end(self, batch, logs={}):
        self.reporter.broadcast(logs.get(self.metric_name, 0))


class KerasEpochEnd(tf.keras.callbacks.Callback):
    """A Keras callback reporting a specified `metric` at the end of an epoch
    to the maggy experiment driver.

    `val_loss` is always available as a metric, and optionally `val_acc` (if
    accuracy monitoring is enabled, that is, accuracy is added to keras model
    metrics). Training metrics are available under the names `loss` and `acc`.
    Default is validation loss (`val_loss`).

    Example usage:

    >>> from maggy.callbacks import KerasBatchEnd
    >>> callbacks = [KerasBatchEnd(reporter, metric='val_acc')]
    """

    def __init__(self, reporter, metric="val_loss"):
        super().__init__()
        self.metric_name = metric
        self.reporter = reporter

    def on_epoch_end(self, epoch, logs={}):
        self.reporter.broadcast(logs.get(self.metric_name, 0), epoch)
