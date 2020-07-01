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

def get_list_of_layer_names(config_dict):
    """
        Return a list of layer names.

        :param config_dict: the config dictionary of a Keras functional model, i.e., the output of model.get_config()
        :type config_dict: dict

        :rtype: `list`
    """

    list_of_names = [layer['name'] for layer in config_dict['layers']]
    return list_of_names

def get_dict_of_inbound_layers_mapping(config_dict):
    """
    Return a dictionary with layer names as keys and a list of their inbound layers as values.

    :param config_dict: the config dictionary of a Keras functional model, i.e., the output of model.get_config()
    :type config_dict: dict

    :rtype: `dict`
    """

    dict_of_inbound_layers = {}
    for layer in config_dict['layers']:
        list_of_inbound_layers = []
        name = layer['name']
        if len(layer['inbound_nodes']) > 0: # because some layers, such as input layers, do not have any inbound_nodes
            for inbound_layer in layer['inbound_nodes'][0]:
                list_of_inbound_layers.append(inbound_layer[0])
        dict_of_inbound_layers[name] = list_of_inbound_layers

    return dict_of_inbound_layers


def get_layers_for_removal(starting_layer, ending_layer, inbound_layers_mapping_dict, layers_for_removal=[]):
    """
    Return a list containing the names of all the layers that have to be removed, i.e., all the layers
    in a module.

    :param starting_layer: name of the starting layer of the module, e.g., a mixed (concat) layer
    :type starting_layer: str
    :param ending_layer: name of the ending layer of the module, e.g., a mixed (concat) layer
    :type ending_layer: str
    :param layers_mapping_dict: a dictionary of inbound layers mapping, i.e. the output of
    get_dict_of_inbound_layers_mapping() 
    :type layers_mapping_dict: dict
    :param layers_for_removal: a list of layers to be removed, for the recursive calls
    :type layers_for_removal: list
    :param config_dict: the config dictionary of a Keras functional model, i.e., the output of model.get_config()
    :type config_dict: dict
    :rtype: `dict`
    """

    if ending_layer == starting_layer:
        return
    
    for inbound_layer in inbound_layers_mapping_dict[ending_layer]:
        get_layers_for_removal(starting_layer, inbound_layer, inbound_layers_mapping_dict, layers_for_removal)

    inbound_layers_mapping_dict.pop(ending_layer) # not sure how this will change the state
    layers_for_removal.append(ending_layer)

    return layers_for_removal



