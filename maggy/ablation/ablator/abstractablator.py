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

from abc import ABC, abstractmethod


class AbstractAblator(ABC):
    def __init__(self, ablation_study, final_store):
        self.ablation_study = ablation_study
        self.final_store = final_store
        self.trial_buffer = []

    @abstractmethod
    def get_number_of_trials(self):
        """
        If applicable, calculate and return the total number of trials of the ablation experiment.
        Make sure to also include the base (reference) trial in the count.

        :return: total number of trials of the ablation study experiment
        :rtype: int
        """
        pass

    @abstractmethod
    def get_dataset_generator(self, ablated_feature, dataset_type="tfrecord"):
        """
        Create and return a dataset generator function based on the ablation policy to be used in a trial.
        The returned function will be executed on the executor per each trial.

        :param ablated_feature: the name of the feature to be excluded from the training dataset.
            Must match a feature name in the corresponding feature group in the feature store.
        :type ablated_feature: str
        :param dataset_type: type of the dataset. For now, we only support 'tfrecord'.
        :return: A function that generates a TFRecordDataset
        :rtype: function
        """
        pass

    @abstractmethod
    def get_model_generator(self, ablated_layer):
        pass

    @abstractmethod
    def initialize(self):
        """
        Initialize the ablation study experiment by generating a number of trials. Depending on the ablation policy,
        this method might generate all the trials (e.g. as in LOCO), or generate a number of trials to warm-start the
        experiment. The trials should be added to `trial_buffer` in form of `Trial` objects.
        """
        pass

    @abstractmethod
    def get_trial(self, ablation_trial=None):
        """
        Return a `Trial` to be assigned to an executor, or `None` if there are no trials remaining in the experiment.
        The trial should contain a dataset generator and a model generator.
        Depending on the ablator policy, the trials could come from a list (buffer) of pre-made trials,
        or generated on the fly.

        :rtype: `Trial` or `None`
        """
        pass

    @abstractmethod
    def finalize_experiment(self, trials):
        """
        This method will be called before finishing the experiment. Developers can implement this method
        e.g. for cleanup or extra logging.
        """
        pass

    def name(self):
        return str(self.__class__.__name__)
