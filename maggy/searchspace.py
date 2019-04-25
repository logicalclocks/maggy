import json


class Searchspace(object):
    """Create an instance of `Searchspace` from keyword arguments.

    The keyword arguments specify name-values pairs for the hyperparameters,
    where values are tuples of the form (type, list). Type is a string with
    one of the following values:

        - DOUBLE
        - INTEGER
        - DISCRETE
        - CATEGORICAL

    And the list in the tuple specifies either two values only, the start
    and end point of of the feasible interval for DOUBLE and INTEGER,
    or the discrete possible values for the types DISCRETE and CATEGORICAL.

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
            raise ValueError(
                "Hyperparameter name is reserved: {}".format(name))

        if isinstance(value, tuple) or isinstance(value, list):

            if len(value) != 2:
                raise ValueError("Hyperparameter tuple has to be of length "
                                 "two and format (type, list): {0}, {1}".format(name, value))

            param_type = value[0].upper()
            param_values = value[1]

            if param_type in [Searchspace.DOUBLE,
                              Searchspace.INTEGER,
                              Searchspace.DISCRETE,
                              Searchspace.CATEGORICAL]:

                if len(param_values) == 0:
                    raise ValueError("Hyperparameter feasible region list "
                                     "cannot be empty: {0}, {1}".format(name, param_values))

                if param_type in [Searchspace.DOUBLE,
                                  Searchspace.INTEGER]:
                    assert len(param_values) == 2, ("For DOUBLE or INTEGER type parameters, list "
                                                    "can only contain upper and lower bounds: {0}, {1}".format(
                                                        name, param_values
                                                    ))

                    if param_type == Searchspace.DOUBLE:
                        if (type(param_values[0]) not in [int, float] or
                                type(param_values[1]) not in [int, float]):
                            raise ValueError("Hyperparameter boundaries for type DOUBLE need to be integer "
                                             "or float: {}".format(name))
                    elif param_type == Searchspace.INTEGER:
                        if (type(param_values[0]) != int or
                                type(param_values[1]) != int):
                            raise ValueError("Hyperparameter boundaries for type INTEGER need to be integer: "
                                             "{}".format(name))

                    assert param_values[0] < param_values[1], ("Lower bound {0} must be "
                                                               "less than upper bound {1}: {2}".format(
                                                                   param_values[0], param_values[1], name
                                                               ))

                self._hparam_types[name] = param_type
                setattr(self, name, value[1])
            else:
                raise ValueError("Hyperparameter type is not of type DOUBLE, "
                                 "INTEGER, DISCRETE or CATEGORICAL: {}".format(name))

        else:
            raise ValueError("Value is not an appropriate tuple: {}"
                             .format(name)
                             )

        print("Hyperparameter added: {}".format(name))

    def to_dict(self):
        """Return the hyperparameters as a Python dictionary.

        :return: A dictionary with hyperparameter names as keys. The values are
            the hyperparameter values.
        :rtype: dict
        """
        return {n: (self._hparam_types[n], getattr(self, n)) for n in self._hparam_types.keys()}

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

    def __contains__(self, name):
        return name in self._hparam_types

    def __str__(self):
        return json.dumps(self.to_dict(), sort_keys=True)
