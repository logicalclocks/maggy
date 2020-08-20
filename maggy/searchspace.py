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
import random

import numpy as np


class Searchspace(object):
    """Create an instance of `Searchspace` from keyword arguments.

    A searchspace is essentially a set of key value pairs, defining the
    hyperparameters with a name, type and a feasible interval. The keyword
    arguments specify name-values pairs for the hyperparameters,
    where values are tuples of the form (type, list). Type is a string with
    one of the following values:

        - DOUBLE
        - INTEGER
        - DISCRETE
        - CATEGORICAL

    And the list in the tuple specifies either two values only, the start
    and end point of of the feasible interval for DOUBLE and INTEGER,
    or the discrete possible values for the types DISCRETE and CATEGORICAL.

    Sample usage:

    >>> # Define Searchspace
    >>> from maggy import Searchspace
    >>> # The searchspace can be instantiated with parameters
    >>> sp = Searchspace(kernel=('INTEGER', [2, 8]), pool=('INTEGER', [2, 8]))
    >>> # Or additional parameters can be added one by one
    >>> sp.add('dropout', ('DOUBLE', [0.01, 0.99]))

    The `Searchspace` object can also be initialized from a python dictionary:

    >>> sp_dict = sp.to_dict()
    >>> sp_new = Searchspace(**sp_dict)

    The parameter names are added as attributes of `Searchspace` object,
    so they can be accessed directly with the dot notation
    `searchspace._name_`.
    """

    DOUBLE = "DOUBLE"
    INTEGER = "INTEGER"
    DISCRETE = "DISCRETE"
    CATEGORICAL = "CATEGORICAL"

    def __init__(self, **kwargs):
        self._hparam_types = {}
        self._names = []
        for name, value in kwargs.items():
            self.add(name, value)

    def add(self, name, value):
        """Adds {name, value} pair to hyperparameters.

        :param name: Name of the hyperparameter
        :type name: str
        :param value: A tuple of the parameter type and its feasible region
        :type value: tuple
        :raises ValueError: Hyperparameter name is reserved
        :raises ValueError: Hyperparameter feasible region in wrong format
        """
        if getattr(self, name, None) is not None:
            raise ValueError("Hyperparameter name is reserved: {}".format(name))

        if isinstance(value, tuple) or isinstance(value, list):

            if len(value) != 2:
                raise ValueError(
                    "Hyperparameter tuple has to be of length "
                    "two and format (type, list): {0}, {1}".format(name, value)
                )

            param_type = value[0].upper()
            param_values = value[1]

            if param_type in [
                Searchspace.DOUBLE,
                Searchspace.INTEGER,
                Searchspace.DISCRETE,
                Searchspace.CATEGORICAL,
            ]:

                if len(param_values) == 0:
                    raise ValueError(
                        "Hyperparameter feasible region list "
                        "cannot be empty: {0}, {1}".format(name, param_values)
                    )

                if param_type in [Searchspace.DOUBLE, Searchspace.INTEGER]:
                    assert len(param_values) == 2, (
                        "For DOUBLE or INTEGER type parameters, list "
                        "can only contain upper and lower bounds: {0}, {1}".format(
                            name, param_values
                        )
                    )

                    if param_type == Searchspace.DOUBLE:
                        if type(param_values[0]) not in [int, float] or type(
                            param_values[1]
                        ) not in [int, float]:
                            raise ValueError(
                                "Hyperparameter boundaries for type DOUBLE need to be integer "
                                "or float: {}".format(name)
                            )
                    elif param_type == Searchspace.INTEGER:
                        if type(param_values[0]) != int or type(param_values[1]) != int:
                            raise ValueError(
                                "Hyperparameter boundaries for type INTEGER need to be integer: "
                                "{}".format(name)
                            )

                    assert param_values[0] < param_values[1], (
                        "Lower bound {0} must be "
                        "less than upper bound {1}: {2}".format(
                            param_values[0], param_values[1], name
                        )
                    )

                self._hparam_types[name] = param_type
                setattr(self, name, value[1])
                self._names.append(name)
            else:
                raise ValueError(
                    "Hyperparameter type is not of type DOUBLE, "
                    "INTEGER, DISCRETE or CATEGORICAL: {}".format(name)
                )

        else:
            raise ValueError("Value is not an appropriate tuple: {}".format(name))

        print("Hyperparameter added: {}".format(name))

    def to_dict(self):
        """Return the hyperparameters as a Python dictionary.

        :return: A dictionary with hyperparameter names as keys. The values are
            the hyperparameter values.
        :rtype: dict
        """
        return {
            n: (self._hparam_types[n], getattr(self, n))
            for n in self._hparam_types.keys()
        }

    def names(self):
        """Returns the dictionary with the names and types of all
        hyperparameters.

        :return: Dictionary of hyperparameter names, with types as value
        :rtype: dict
        """
        return self._hparam_types

    def get(self, name, default=None):
        """Returns the value of `name` if it exists, else `default`."""
        if name in self._hparam_types:
            return getattr(self, name)

        return default

    def get_random_parameter_values(self, num):
        """Generate random parameter dictionaries, e.g. to be used for initializing an optimizer.

        :param num: number of random parameter dictionaries to be generated.
        :type num: int
        :raises ValueError: `num` is not an int.
        :return: a list containing parameter dictionaries
        :rtype: list
        """
        return_list = []
        for _ in range(num):
            params = {}
            for name, value in self.names().items():
                feasible_region = self.get(name)
                if value == Searchspace.DOUBLE:
                    params[name] = random.uniform(
                        feasible_region[0], feasible_region[1]
                    )
                elif value == Searchspace.INTEGER:
                    params[name] = random.randint(
                        feasible_region[0], feasible_region[1]
                    )
                elif value == Searchspace.DISCRETE:
                    params[name] = random.choice(feasible_region)
                elif value == Searchspace.CATEGORICAL:
                    params[name] = random.choice(feasible_region)
            return_list.append(params)

        return return_list

    def __iter__(self):
        self._returned = self._names.copy()
        return self

    def __next__(self):
        # if list not empty
        if self._returned:
            # pop from left and get parameter tuple
            name = self._returned.pop(0)
            return {
                "name": name,
                "type": self._hparam_types[name],
                "values": self.get(name),
            }
        else:
            raise StopIteration

    def items(self):
        """Returns a sorted iterable over all hyperparameters in the searchspace.

        Allows to iterate over the hyperparameters in a searchspace. The parameters
        are sorted in the order of which they were added to the searchspace by the user.

        :return: an iterable of the searchspace
        :type: Searchspace
        """
        # for consistency and serves mainly as syntactic sugar
        return self

    def keys(self):
        """Returns a sorted iterable list over the names of hyperparameters in
        the searchspace.

        :return: names of hyperparameters as a list of strings
        :type: list
        """
        return self._names

    def values(self):
        """Returns a sorted iterable list over the types and feasible intervals of
        hyperparameters in the searchspace.

        :return: types and feasible interval of hyperparameters as tuple
        :type: tuple
        """
        return [(self._hparam_types[name], self.get(name)) for name in self._names]

    def __contains__(self, name):
        return name in self._hparam_types

    def __str__(self):
        return json.dumps(self.to_dict(), sort_keys=True)

    def json(self):
        return json.dumps(self.to_dict(), sort_keys=True)

    def transform(self, hparams, normalize_categorical=False):
        """Transforms array of hypeparameters for one trial.

        +--------------+-----------------------------------------------------+
        | Hparam Type  | Transformation                                      |
        +==============+=====================================================+
        | DOUBLE       | Max-Min Normalization                               |
        +--------------+-----------------------------------------------------+
        | INTEGER      | Max-Min Normalization                               |
        +--------------+-----------------------------------------------------+
        | CATEGORICAL  | Encoding: index in list + opt. Max-Min Normalization|
        +--------------+-----------------------------------------------------+

        :param hparams: hparams in original representation for one trial
        :type hparams: 1D np.ndarray
        :param normalize_categorical: If True, the encoded categorical hparam is also max-min normalized between 0 and 1
        `inverse_transform()` must use the same value for this parameter
        :type normalize_categorical: bool
        :return: transformed hparams
        :rtype: np.ndarray[np.float]
        """
        transformed_hparams = []
        # loop through hparams
        for hparam, hparam_spec in zip(hparams, self.items()):
            if hparam_spec["type"] == "DOUBLE":
                normalized_hparam = Searchspace._normalize_scalar(
                    hparam_spec["values"], hparam
                )
                transformed_hparams.append(normalized_hparam)
            elif hparam_spec["type"] == "INTEGER":
                normalized_hparam = Searchspace._normalize_integer(
                    hparam_spec["values"], hparam
                )
                transformed_hparams.append(normalized_hparam)
            elif hparam_spec["type"] == "CATEGORICAL":
                encoded_hparam = Searchspace._encode_categorical(
                    hparam_spec["values"], hparam
                )
                if normalize_categorical:
                    encoded_hparam = Searchspace._normalize_integer(
                        [0, len(hparam_spec["values"]) - 1], encoded_hparam
                    )
                transformed_hparams.append(encoded_hparam)
            else:
                raise NotImplementedError("Not Implemented other types yet")

        return transformed_hparams

    def inverse_transform(self, transformed_hparams, normalize_categorical=False):
        """Returns array of hparams in same representation as specified when instantiated

        :param transformed_hparams: hparams in transformed representation for one trial
        :type transformed_hparams: 1D np.ndarray
        :param normalize_categorical: If True, the encoded categorical hparam was also max-min normalized between 0 and 1
        `transform()` must use the same value for this parameter
        :type normalize_categorical: bool
        :return: transformed hparams
        :rtype: np.ndarray
        """
        hparams = []
        for hparam, hparam_spec in zip(transformed_hparams, self.items()):
            if hparam_spec["type"] == "DOUBLE":
                value = Searchspace._inverse_normalize_scalar(
                    hparam_spec["values"], hparam
                )
                hparams.append(value)
            elif hparam_spec["type"] == "INTEGER":
                value = Searchspace._inverse_normalize_integer(
                    hparam_spec["values"], hparam
                )
                hparams.append(value)
            elif hparam_spec["type"] == "CATEGORICAL":
                if normalize_categorical:
                    value = Searchspace._inverse_normalize_integer(
                        [0, len(hparam_spec["values"]) - 1], hparam
                    )
                    value = Searchspace._decode_categorical(
                        hparam_spec["values"], value
                    )
                else:
                    value = Searchspace._decode_categorical(
                        hparam_spec["values"], hparam
                    )
                hparams.append(value)
            else:
                raise NotImplementedError("Not Implemented other types yet")

        return hparams

    @staticmethod
    def _encode_categorical(choices, value):
        """Encodes category to integer. The encoding is the list index of the category

        :param choices: possible values of the categorical hparam
        :type choices: list
        :param value: category to encode
        :type value: str
        :return: encoded category
        :rtype: int
        """
        return choices.index(value)

    @staticmethod
    def _decode_categorical(choices, encoded_value):
        """Decodes integer to corresponding category value

        :param choices: possible values of the categorical hparam
        :type choices: list
        :param encoded_value: encoding of category
        :type encoded_value: int
        :return: category value
        :rtype: str
        """
        encoded_value = int(
            encoded_value
        )  # it is possible that value gets casted to np.float by numpy
        return choices[encoded_value]

    @staticmethod
    def _normalize_scalar(bounds, scalar):
        """Returns max-min normalized scalar

        :param bounds: list containing lower and upper bound, e.g.: [-3,3]
        :type bounds: list
        :param scalar: scalar value to be normalized
        :type scalar: float
        :return: normalized scalar
        :rtype: float
        """
        scalar = float(scalar)
        scalar = (scalar - bounds[0]) / (bounds[1] - bounds[0])
        scalar = np.minimum(1.0, scalar)
        scalar = np.maximum(0.0, scalar)
        return scalar

    @staticmethod
    def _inverse_normalize_scalar(bounds, normalized_scalar):
        """Returns inverse normalized scalar

        :param bounds: list containing lower and upper bound, e.g.: [-3,3]
        :type bounds: list
        :param normalized_scalar: normalized scalar value
        :type normalized_scalar: float
        :return: original scalar
        :rtype: float
        """
        normalized_scalar = float(normalized_scalar)
        normalized_scalar = normalized_scalar * (bounds[1] - bounds[0]) + bounds[0]
        return normalized_scalar

    @staticmethod
    def _normalize_integer(bounds, integer):
        """
        :param bounds: list containing lower and upper bound, e.g.: [-3,3]
        :type bounds: list
        :param integer: value to be normalized
        :type normalized_scalar: int
        :return: normalized value between 0 and 1
        :rtype: float
        """

        integer = int(integer)
        return Searchspace._normalize_scalar(bounds, integer)

    @staticmethod
    def _inverse_normalize_integer(bounds, scalar):
        """Returns inverse normalized scalar

        :param bounds: list containing lower and upper bound, e.g.: [-3,3]
        :type bounds: list
        :param normalized_scalar: normalized scalar value
        :type normalized_scalar: float
        :return: original integer
        :rtype: int
        """

        x = Searchspace._inverse_normalize_scalar(bounds, scalar)
        return int(np.round(x))

    @staticmethod
    def dict_to_list(hparams):
        """Transforms dict of hparams to list representation ( for one hparam config )

        example:
            {'x': -3.0, 'y': 3.0, 'z': 'green'} to [-3.0, 3.0, 'green']

        :param hparams: hparams in dict representation
        :type hparams: dict
        :return: hparams in list representation
        :rtype: list
        """
        return list(hparams.values())

    def list_to_dict(self, hparams):
        """Transforms list of hparams to dict representation ( for one hparam config )

        example:
            [-3.0, 3.0, 'green'] to {'x': -3.0, 'y': 3.0, 'z': 'green'}

        :param hparams: hparams in list representation
        :type hparams: list
        :return: hparams in dict representation
        :rtype: dict
        """
        hparam_names = self.keys()
        if len(hparam_names) != len(hparams):
            raise ValueError(
                "hparam_names and hparams have to have same length (and order!)"
            )

        hparam_dict = {
            hparam_name: hparam for hparam_name, hparam in zip(hparam_names, hparams)
        }
        return hparam_dict
