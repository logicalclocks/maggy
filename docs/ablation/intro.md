# Quick Start

Ablation studies have become best practice in evaluating model architectures, as they provide insights into the relative 
contribution of the different architectural and regularization components to the performance of models. 
An ablation study consists of several trials, where one trial could be, 
e.g., removing the last convolutional layer of a CNN model, retraining the model, 
and observing the resulting performance. 
However, as machine learning architectures become ever deeper and data sizes keep growing, 
there is an explosion in the number of different architecture combinations that need to be evaluated to understand 
their relative performance. 

Maggy provides a declarative way to define ablation experiments for model architectures 
and training datasets, in a way that eliminates the need for maintaining redundant copies of code for an ablation study.
Furthermore, our framework enables parallel execution of ablation trials without requiring the developers to 
modify their code, which leads to shorter study times and better resource utilization. 

## Simple Example

In order to use Maggy for Ablation studies, first we need to define a model generator function.
```py
from maggy.ablation import AblationStudy

ablation_study = AblationStudy('titanic_train_dataset', training_dataset_version=1,
                              label_name='survived')
```

### Add the features to ablate

We perform feature ablation by including features in our AblationStudy instance. Including a feature means that there 
will be a trial where the model will be trained without that feature. 
In other words, you include features in the ablation study so that they will be excluded from the training dataset.

In this example, we have the following features in our training dataset:
```
['age', 'fare', 'parch', 'pclass', 'sex', 'sibsp', 'survived']
```

You can include features using ```features.include()``` method of your AblationStudy instance, 
by passing the names of the features, either separately or as a list of strings:

```py
#include features one by one
ablation_study.features.include('pclass')

# include a list of features
list_of_features = ['fare', 'sibsp']
ablation_study.features.include(list_of_features)
```

### Add the model layers to ablate

By model ablation we mean removing the components of the model, retraining and observing the resulting performance. 
Depending on which component of the model you want to ablate, we could have different types of model ablation, 
but one thing they all share in common is that we should have one base model in order to compare the other 
models with it. So we should define a base model generator function that returns a ```tf.keras.Model```.

Maybe the simplest type of model ablation that can be performed on a sequential deep learning model is to 
remove some of its layers, so let’s just do that. We call this layer ablation. 
In Keras, when you are adding layers to your ```Sequential``` model, 
you can provide a name argument with a custom name. 
The Maggy ablator then uses these names to identify and remove the layers. 
Then, for each trial, the ablator returns a corresponding model generator function that differs from the base 
model generator in terms of its layers.

By the way, if you do not provide a name argument while adding a layer, that layer’s name will be prefixed by its 
layer class name, followed by an incremental number, e.g., dense_1.

In the following cell, we define a base model generator function that once executed will 
return a ```Sequential``` model:
```py
def base_model_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, name='my_dense_two', activation='relu'))
    model.add(tf.keras.layers.Dense(32, name='my_dense_three', activation='relu'))
    model.add(tf.keras.layers.Dense(32, name='my_dense_four', activation='relu'))
    model.add(tf.keras.layers.Dense(2, name='my_dense_sigmoid', activation='sigmoid'))
    # output layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    return model

ablation_study.model.set_base_model_generator(base_model_generator)

```

Adding layers to your ablation study is easy - just pass their names to ```model.layers.include()```, 
or pass a list of strings of the names. Of course, these names should match the names you define in your base model 
generator function:
```py
# include some single layers in the ablation study

ablation_study.model.layers.include('my_dense_two', 
                                    'my_dense_three', 
                                    'my_dense_four', 
                                    'my_dense_sigmoid'
                                    )
```

For smaller models, it might make sense to add the layers to the model one-by-one. 
However, in many well-known neural network architectures, such as Transformers, 
we have tens or hundreds of layers that sometimes come 
in blocks or modules, and are usually generated using constructs like for loops.

In Maggy, you can easily include such layers in your ablation study experiment, 
using ```model.layers.include_groups()``` method of your AblationStudy instance. 
You can either pass it a list of layers that should be regarded as a single layer group, 
or provide it with a prefix argument:

```py
# add a layer group using a list
ablation_study.model.layers.include_groups(['my_dense_two', 'my_dense_four'])

# add a layer group using a prefix
ablation_study.model.layers.include_groups(prefix='my_dense')
```

### Write the training logic

Now the only thing you need to do is to write your training code in a Python function. 
You can name this function whatever you wish, but we will refer to it as the training function. 
The model_function and dataset_function used in the code are generated by the ablator per each trial, 
and you should call them in your code. This is your run-of-the-mill TensorFlow/Keras code:

```py
# wrap your code in a Python function
def training_fn(dataset_function, model_function):
    import tensorflow as tf
    epochs = 5
    batch_size = 10
    tf_dataset = dataset_function(epochs, batch_size)
    model = model_function()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss='binary_crossentropy',
             metrics=['accuracy'])
    
    history = model.fit(tf_dataset, epochs=5, steps_per_epoch=30, verbose=0)
    return float(history.history['accuracy'][-1])
```
And, lagom! Lagom is a Swedish word that means “just the right amount”, and that is how Maggy uses your resources to 
for parallel trials. You should provide lagom with an ablator. So far, we have implemented the most natural ablator of 
them all: LOCO, which stands for “Leave One Component Out”. This ablator will generate one trial per each component 
included in the ablation study. However, Maggy’s developer API allows you to define your own ablator, 
in case you want to get creative.

You can also set a name for your experiment so that you can keep history or track its progress in Hopsworks.

Let’s lagom our experiment!

```py
# Create a config for lagom
from maggy.experiment_config import AblationConfig
from maggy.experiment import experiment

config = AblationConfig(name="Titanic-LOCO", ablation_study=ablation_study, ablator="loco", description="", hb_interval=1)
# launch the experiment
result = experiment.lagom(train_fn=training_fn, config=config)
```