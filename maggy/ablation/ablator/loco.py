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

from maggy.ablation.ablator import AbstractAblator
from maggy.core.exceptions import NotSupportedError
from maggy.core.exceptions import BadArgumentsError
from hops import featurestore
from maggy.trial import Trial
import json


class LOCO(AbstractAblator):
    def __init__(self, ablation_study, final_store):
        super().__init__(ablation_study, final_store)
        self.base_dataset_generator = self.get_dataset_generator(ablated_feature=None)

    def get_number_of_trials(self):
        # the final ' + 1 ' is for the base (reference) trial with all the components
        return (
            len(self.ablation_study.features.included_features)
            + len(self.ablation_study.model.layers.included_layers)
            + len(self.ablation_study.model.layers.included_groups)
            + len(self.ablation_study.model.custom_model_generators)
            + len(self.ablation_study.model.modules)
            + 1
        )

    def get_dataset_generator(
        self, ablated_feature=None, dataset_type="tfrecord"
    ):

        # if available, use the dataset generator provided by the user
        if self.ablation_study.custom_dataset_generator is not (None or False):
            return self.ablation_study.custom_dataset_generator

        # else use a dataset generator from maggy
        training_dataset_name = self.ablation_study.hops_training_dataset_name
        training_dataset_version = self.ablation_study.hops_training_dataset_version
        label_name = self.ablation_study.label_name

        if dataset_type == "tfrecord":

            def dataset_generator(num_epochs, batch_size):
                from maggy.ablation.utils import tensorflowdatasets
                return tensorflowdatasets.ablate_feature_and_create_tfrecord_dataset_from_featurestore(ablated_feature=ablated_feature, 
                training_dataset_name=training_dataset_name, training_dataset_version=training_dataset_version,
                label_name=label_name, num_epochs=num_epochs, batch_size=batch_size)
            return dataset_generator

        else:
            raise NotSupportedError(
                "dataset type",
                dataset_type,
                "Use 'tfrecord' or write your own custom dataset generator.",
            )

    def get_model_generator(self, layer_identifier=None, custom_model_generator=None, ablation_type=None,
    starting_layer=None, ending_layer=None):
        
        base_model_generator = self.ablation_study.model.base_model_generator

        # 1 - for the base trial, return the base_model_generator
        if ablation_type=='base':
            return base_model_generator
        
        # 2 - if this trial relates to a custom model, then return the provided custom model generator
        elif ablation_type=='custom_model' and custom_model_generator is not None:
            return custom_model_generator[0]

        # 3 - for a layer ablation trial of a sequential model, return a model_generator()
        elif ablation_type=='layer':
            def model_generator():
                from maggy.ablation.utils import sequentialmodel
                return sequentialmodel.model_generator_for_layer_ablation(layer_identifier, base_model_generator)
            return model_generator

        # 4 - for a module ablation trial of a functional model, return a model generator
        elif ablation_type=='module':
            def model_generator():
                from maggy.ablation.utils import functionalmodel
                return functionalmodel.model_generator_for_module_ablation(starting_layer, ending_layer, base_model_generator)
            return model_generator


    def initialize(self):
        """
        Prepares all the trials for LOCO policy (Leave One Component Out).
        In total `n+1` trials will be generated where `n` is equal to the number of components
        (e.g. features, layers, and modules) that are included in the ablation study
        (i.e. the components that will be removed one-at-a-time). The first trial will include all the components and
        can be regarded as the base for comparison.
        """
        
        # 0 - add first trial with all the components (base/reference trial)
        self.trial_buffer.append(
            Trial(self.create_trial_dict(ablation_type='base'), trial_type="ablation")
        )

        # generate remaining trials based on the ablation study configuration:
        # 1 - generate feature ablation trials
        for feature in self.ablation_study.features.included_features:
            self.trial_buffer.append(
                Trial(
                    self.create_trial_dict(ablated_feature=feature, ablation_type='feature'),
                    trial_type="ablation",
                )
            )

        # 2 - generate single-layer ablation trials
        for layer in self.ablation_study.model.layers.included_layers:
            self.trial_buffer.append(
                Trial(
                    self.create_trial_dict(layer_identifier=layer, ablation_type='layer'),
                    trial_type="ablation",
                )
            )

        # 3 - generate layer-groups ablation trials
        # each element of `included_groups` is a frozenset of a set, so we cast again to get a set
        # why frozensets in the first place? because a set can only contain immutable elements
        # hence elements (layer group identifiers) are frozensets

        for layer_group in self.ablation_study.model.layers.included_groups:
            self.trial_buffer.append(
                Trial(
                    self.create_trial_dict(layer_identifier=set(layer_group), ablation_type='layer'),
                    trial_type="ablation",
                )
            )

        # 4 - generate ablation trials based on custom model generators

        for custom_model_generator in self.ablation_study.model.custom_model_generators:
            self.trial_buffer.append(
                Trial(
                    self.create_trial_dict(
                        custom_model_generator=custom_model_generator, ablation_type='custom_model'
                    ),
                    trial_type="ablation",
                )
            )
        
        # 5 - generate module ablation trials
        for module in self.ablation_study.model.modules:
            starting_layer, ending_layer = module
            self.trial_buffer.append(
                Trial(
                    self.create_trial_dict(
                        starting_layer=starting_layer,
                        ending_layer=ending_layer,
                        ablation_type='module',
                    ),
                    trial_type='ablation'
                )
            )        

    def get_trial(self, trial=None):
        if self.trial_buffer:
            return self.trial_buffer.pop()
        else:
            return None

    def finalize_experiment(self, trials):
        return

    def create_trial_dict(
        self, ablation_type=None, ablated_feature=None, layer_identifier=None, custom_model_generator=None, 
        starting_layer=None, ending_layer=None,
    ):
        """
        Creates a trial dictionary that can be used for creating a Trial instance.

        :param ablation_type: a string representing type of ablation trial 
        ('feature', 'layer', 'module', 'custom_model', or 'base')
        :param ablated_feature: a string representing the name of a feature, or None
        :param layer_identifier: A string representing the name of a single layer, or a set representing a layer group.
        If the set has only one element, it is regarded as a prefix, so all layers with that prefix in their names
        would be regarded as a layer group. Otherwise, if the set has more than one element, the layers with
        names corresponding to those elements would be regarded as a layer group.
        :return: A trial dictionary that can be passed to maggy.Trial() constructor.
        :rtype: dict
        """

        trial_dict = {}

        # 1 - determine the dataset generation logic
        # 1.1 - if it's a feature ablation trial, prepare the trial_dict
        # using the base model generator and return

        if ablation_type=='feature':
            trial_dict["dataset_function"] = self.get_dataset_generator(ablated_feature, dataset_type="tfrecord")
            trial_dict["ablated_feature"] = ablated_feature
            trial_dict["ablated_layer"] = "None"
            trial_dict["model_function"] = self.ablation_study.model.base_model_generator

            return trial_dict
            
        else:
            trial_dict["dataset_function"] = self.get_dataset_generator()
            trial_dict["ablated_feature"] = "None"

        # 2 - determine the model generation logic
        # 2.1 - no model ablation

        if ablation_type=='base':
            trial_dict["model_function"] = self.get_model_generator(ablation_type='base')
            trial_dict["ablated_layer"] = "None"

        # 2.2 - layer ablation based on base model generator
        elif ablation_type=='layer':
            trial_dict["model_function"] = self.get_model_generator(
                layer_identifier=layer_identifier,
                ablation_type='layer'
            )
            # prepare the string representation of the trial
            if type(layer_identifier) is str:
                trial_dict["ablated_layer"] = layer_identifier
            elif type(layer_identifier) is set:
                if len(layer_identifier) > 1:
                    trial_dict["ablated_layer"] = str(list(layer_identifier))
                elif len(layer_identifier) == 1:
                    trial_dict["ablated_layer"] = "Layers prefixed " + str(
                        list(layer_identifier)[0]
                    )
        # 2.3 - model ablation based on a custom model generator
        elif ablation_type=='custom_model':
            trial_dict["model_function"] = self.get_model_generator(
                custom_model_generator=custom_model_generator,
                ablation_type='custom_model'
            )
            trial_dict["ablated_layer"] = "Custom model: " + custom_model_generator[1]
        
        # 2.4 - module ablation based on base model generator
        elif ablation_type=='module':
            trial_dict['model_function'] = self.get_model_generator(starting_layer=starting_layer, ending_layer=ending_layer, ablation_type='module')
            trial_dict['ablated_layer'] = "All layers between {0} and {1}".format(starting_layer, ending_layer)

        return trial_dict
