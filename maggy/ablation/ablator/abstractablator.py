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

import json
from abc import abstractmethod

from maggy.core import controller


class AbstractAblator(controller.Controller):
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
    def get_next_trial(self, trial=None):
        """
        Return a `Trial` to be assigned to an executor, or `None` if there are no trials remaining in the experiment.
        The trial should contain a dataset generator and a model generator.
        Depending on the ablator policy, the trials could come from a list (buffer) of pre-made trials,
        or generated on the fly.

        :rtype: `Trial` or `None`
        """
        pass

    def finalize(self, result, trials):
        """
        This method will be called before finishing the experiment. Developers
        can override or extend this method e.g. for cleanup or extra logging.
        Maggy expects two values to be returned, a dictionary and a string
        to be printed. The dictionary will be merged with the `result` dict of
        Maggy, persisted and returned to the user. The finalized `trials` can
        be used to compute additional statistics to return.
        The returned string representation of the result should be human readable
        and will be printed and written to the logs.

        :param result: Results of the experiment as dictionary.
        :type result: dict
        :param trials: The finalized trial objects as a list.
        :type trials: list
        :return: result metrics, result string representation
        :rtype: dict, str
        """
        result_dict = {}
        result_str = (
            "\n------ " + self.name() + " Results ------ \n"
            "BEST Config Excludes "
            + json.dumps(result["best_config"])
            + " -- metric "
            + str(result["best_val"])
            + "\n"
            "WORST Config Excludes "
            + json.dumps(result["worst_config"])
            + " -- metric "
            + str(result["worst_val"])
            + "\n"
            "AVERAGE metric -- " + str(result["avg"]) + "\n"
            "Total Job Time " + result["duration_str"] + "\n"
        )
        return result_dict, result_str
