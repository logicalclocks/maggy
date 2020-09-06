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


def model_generator_for_layer_ablation(layer_identifier, base_model_generator):
    import tensorflow as tf
    import json

    base_model = base_model_generator()

    list_of_layers = [base_layer for base_layer in base_model.get_config()["layers"]]
    if type(layer_identifier) is str:
        # ablation of a single layer
        for base_layer in reversed(list_of_layers[1:-1]):
            # the first (input) and last (output) layers should not be considered, hence list_of_layers[1:-1]
            if base_layer["config"]["name"] == layer_identifier:
                list_of_layers.remove(base_layer)
                break
    elif type(layer_identifier) is set:
        # ablation of a layer group - all the layers in the group should be removed together
        if len(layer_identifier) > 1:
            # group of layers (non-prefix)
            for base_layer in reversed(list_of_layers[1:-1]):
                if base_layer["config"]["name"] in layer_identifier:
                    list_of_layers.remove(base_layer)
        elif len(layer_identifier) == 1:
            # layer_identifier is a prefix
            prefix = list(layer_identifier)[0].lower()
            for base_layer in reversed(list_of_layers[1:-1]):
                if base_layer["config"]["name"].lower().startswith(prefix):
                    list_of_layers.remove(base_layer)

    base_json = base_model.to_json()
    new_dict = json.loads(base_json)
    new_dict["config"]["layers"] = list_of_layers
    new_json = json.dumps(new_dict)
    new_model = tf.keras.models.model_from_json(new_json)

    return new_model
