class AblationStudy(object):
    """The `AblationStudy` object is the entry point to define an ablation
    study with maggy. This object can subsequently be passed as an argument
    when the experiment is launched with `experiment.lagom()`.

    Sample usage:

    >>> from maggy.ablation import AblationStudy
    >>> ablation_study = AblationStudy('titanic_train_dataset',
    >>>     label_name='survived')

    Define your study by including layers and features, which should be
    ablated:

    >>> ablation_study.features.include('pclass', 'fare')
    >>> ablation_study.model.layers.include('my_dense_two',
    >>>     'my_dense_three')

    You can also add a layer group using a list:

    >>> ablation_study.model.layers.include_groups(['my_dense_two',
    >>>     'my_dense_four'])

    Or add a layer group using a prefix:

    >>> ablation_study.model.layers.include_groups(prefix='my_dense')

    Next you should define a base model function using the layer and feature
    names you previously specified:

    >>> # you only need to add the `name` parameter to layer initializers
    >>> def base_model_generator():
    >>>     model = tf.keras.Sequential()
    >>>     model.add(tf.keras.layers.Dense(64, activation='relu'))
    >>>     model.add(tf.keras.layers.Dense(..., name='my_dense_two', ...)
    >>>     model.add(tf.keras.layers.Dense(32, activation='relu'))
    >>>     model.add(tf.keras.layers.Dense(..., name='my_dense_sigmoid', ...)
    >>>     # output layer
    >>>     model.add(tf.keras.layers.Dense(1, activation='linear'))
    >>>     return model

    Make sure to include the generator function in the study:

    >>> ablation_study.model.set_base_model_generator(base_model_generator)

    Last but not least you can define your actual training function:

    >>> from maggy import experiment
    >>> from maggy.callbacks import KerasBatchEnd
    ​
    >>> def training_function(dataset_function, model_function, reporter):
    >>>     import tensorflow as tf
    >>>     epochs = 5
    >>>     batch_size = 10
    >>>     tf_dataset = dataset_function(epochs, batch_size)
    >>>     model = model_function()
    >>>     model.compile(optimizer=tf.train.AdamOptimizer(0.001),
    >>>             loss='binary_crossentropy',
    >>>             metrics=['accuracy'])
    >>>     ### Maggy REPORTER
    >>>     callbacks = [KerasBatchEnd(reporter, metric='acc')]
    >>>     history = model.fit(tf_dataset, epochs=5, steps_per_epoch=30)
    >>>     return float(history.history['acc'][-1])

    Lagom the experiment:

    >>> result = experiment.lagom(map_fun=training_function,
    >>>                         experiment_type='ablation',
    >>>                         ablation_study=ablation_study,
    >>>                         ablator='loco',
    >>>                         name='Titanic-LOCO',
    >>>                         hb_interval=5)
    """

    def __init__(
        self, training_dataset_name, training_dataset_version, label_name,
        **kwargs):
        """Initializes the ablation study.

        :param training_dataset_name: Name of the training dataset in the
            featurestore.
        :type training_dataset_name: str
        :param training_dataset_version: Version of the training dataset to be
            used.
        :type training_dataset_version: int
        :param label_name: Name of the target prediction label.
        :type label_name: str
        """
        self.features = Features()
        self.model = Model()
        self.hops_training_dataset_name = training_dataset_name
        self.hops_training_dataset_version = training_dataset_version
        self.label_name = label_name
        self.custom_dataset_generator = kwargs.get('dataset_generator', False)

    def to_dict(self):
        """
        Returns the ablation study configuration as a Python dictionary.

        :return: A dictionary with ablation study configuration parameters as
            keys (i.e. 'training_dataset_name', 'included_features', etc.)
        :rtype: dict
        """
        ablation_dict = {
            'training_dataset_name': self.hops_training_dataset_name,
            'training_dataset_version': self.hops_training_dataset_version,
            'label_name': self.label_name,
            'included_features': list(self.features.included_features),
            'included_layers': list(self.model.layers.included_layers),
            'custom_dataset_generator': True if self.custom_dataset_generator else False,
        }

        return ablation_dict


class Features(object):
    def __init__(self):
        self.included_features = set()

    def include(self, *args):
        """
        Add one or several features of the dataset to the ablation study. Features can be included one by one or
        as a list of strings.
        :param args: Strings or lists of strings, that should match feature names.
        :type args: str or list
        :return:
        """
        for arg in args:
            if type(arg) is list:
                for feature in arg:
                    self._include_single_feature(feature)
            else:
                self._include_single_feature(arg)

    def _include_single_feature(self, feature):
        if type(feature) is str:
            self.included_features.add(feature)
        else:
            raise ValueError(
                "features.include() only accepts strings or lists of strings, "
                "but it received {0} which is of type '{1}'."
                .format(str(feature), type(feature).__name__))

    def exclude(self, *args):
        """
        Exclude one or several features of the dataset from the list of features that have been included in the
        ablation study. Features can be excluded one by one or as a list of strings. This method will check to see
        if the feature has already been included in the study before excluding it.
        :param args: Strings or lists of strings, that should match feature names.
        :type args: str or list
        :return:
        """
        for arg in args:
            if type(arg) is list:
                for feature in arg:
                    self._exclude_single_feature(feature)
            else:
                self._exclude_single_feature(arg)

    def _exclude_single_feature(self, feature):
        if type(feature) is str:
            if feature in self.included_features:
                self.included_features.remove(feature)
                print("Feature '{0}' is excluded from the ablation study.".format(str(feature)))
        else:
            raise ValueError("features.exclude() only accepts strings or lists of strings, "
                             "but it received {0} (of type '{1}')."
                             .format(str(feature), type(feature).__name__))

    def list_all(self):
        for feature in self.included_features:
            print(feature)


class Model(object):
    def __init__(self):
        self.layers = Layers()
        self.base_model_generator = None

    def set_base_model_generator(self, base_model_generator):
        self.base_model_generator = base_model_generator


class Layers(object):
    def __init__(self):
        self.included_layers = set()
        self.included_groups = set()

    def include(self, *args):
        """
        Include layers in the ablations study. Note that the first (input) and the last (output) layer of the base model
        can never be included in the ablation study.
        :param args: Strings or lists of strings, that should match layer names.
        :type args: str or list
        :return:
        """
        for arg in args:
            if type(arg) is list:
                for layer in arg:
                    self._include_single_layer(layer)
            else:
                self._include_single_layer(arg)

    def _include_single_layer(self, layer):
        if type(layer) is str:
            self.included_layers.add(layer)
            # print("included {}".format(layer))  # this still prints even if was duplicate
        else:
            raise ValueError("layers.include() only accepts strings or lists of strings, "
                             "but it received {0} which is of type '{1}'."
                             .format(str(layer), type(layer).__name__))

    def exclude(self, *args):
        for arg in args:
            if type(arg) is list:
                for layer in arg:
                    self._exclude_single_layer(layer)
            else:
                self._exclude_single_layer(arg)

    def _exclude_single_layer(self, layer):
        if type(layer) is str:
            if layer in self.included_layers:
                self.included_layers.remove(layer)
        else:
            raise ValueError("layers.exclude() only accepts strings or lists of strings, "
                             "but it received {0} (of type '{1}')."
                             .format(str(layer), type(layer).__name__))

    def include_groups(self, *args, prefix=None):
        """
        Adds a group of layers that should be removed from the model together. The groups are specified either
        by being passed as a list of layer names (strings), or a string as a common prefix of their layer names.
        Each list of strings would result in a single grouping.
        :param prefix: A string that is a prefix of the names of a group of layers in the base model.
        :type prefix: str
        :param args: Lists of strings (layer names) to indicate groups of layers. The length of the list should be
        greater than one - it does not make sense to have a group of layers that consists of only one layer.
        :type args: list
        """
        if prefix is not None:
            if type(prefix) is str:
                self.included_groups.add(frozenset([prefix]))
            else:
                raise ValueError("`prefix` argument of layers.include_groups() should either be "
                                 "a `NoneType` or a `str`, but it received {0} (of type '{1}'."
                                 .format(str(prefix), type(prefix).__name__))

        for arg in args:
            if type(arg) is list and len(arg) > 1:
                self.included_groups.add(frozenset(arg))
            elif type(arg) is list and len(arg) == 1:
                raise ValueError("layers.include_groups() received a list ( {0} ) "
                                 "with only one element: Did you want to include a single layer in "
                                 "your ablation study? then you should use layers.include()."
                                 .format(str(arg)))
            else:
                raise ValueError("layers.include_groups() only accepts a prefix string, "
                                 "or lists (with more than one element) "
                                 "of strings, but it received {0} (of type '{1}')."
                                 .format(str(arg), type(arg).__name__))

    def exclude_groups(self, *args, prefix=None):
        """
        Removes a group of layers from being included in the ablation study. The groups are specified
        either by being passed as a list of layer names, or by passing a `prefix` argument, denoting a prefix shared by
        all layer names in a group.
        :param prefix: A string that is a prefix of the names of a group that is already included in the ablation study.
        :type prefix: str
        :param args: Lists of strings (layer names) that correspond to groups that are already
        included in the ablation study.
        :type args: list
        """

        if prefix is not None:
            if type(prefix) is str:
                if frozenset([prefix]) in self.included_groups:
                    self.included_groups.remove(frozenset([prefix]))
            else:
                raise ValueError("`prefix` argument of layers.exclude_groups() should either be "
                                 "a `NoneType` or a `str`, but it received {0} (of type '{1}'."
                                 .format(str(prefix), type(prefix).__name__))

        for arg in args:
            if type(arg) is list and len(arg) > 1:
                if frozenset(arg) in self.included_groups:
                    self.included_groups.remove(frozenset(arg))
            else:
                raise ValueError("layers.exclude_groups() only accepts a prefix string, or "
                                 "lists (with more than one element) "
                                 "of strings, but it received {0} (of type '{1}')."
                                 .format(str(arg), type(arg).__name__))

    def print_all(self):
        """
        Prints all single layers that are included in the current ablation study configuration.
        """
        if len(self.included_layers) > 0:
            print("Included single layers are: \n")
            for layer in self.included_layers:
                print(layer)
        else:
            print("There are no single layers in this ablation study configuration.")

    def print_all_groups(self):
        """
        Prints all layer groups that are included in the current ablation study configuration.
        """
        if len(self.included_groups) > 0:
            print("Included layer groups are: \n")
            for layer_group in self.included_groups:
                if len(layer_group) > 1:
                    print("--- Layer group " + str(list(layer_group)))
                elif len(layer_group) == 1:
                    print('---- All layers prefixed "' + str(list(layer_group)[0]) + '"')
        else:
            print("There are no layer groups in this ablation study configuration.")
