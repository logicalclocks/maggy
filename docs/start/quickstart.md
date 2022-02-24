# Quickstart

The programming model consists of wrapping the code containing the model training logic inside a function. 
Inside that wrapper function provide all imports and parts that make up your experiment.

## What can I do with Maggy?

Maggy can be used for three different ML experiments: Distributed Hyperparameter Optimization (HPO), 
Distributed Ablation Study and Distributed Training (DT), their availability depends on which framework
(either TensorFlow or PyTorch) and which platform (Hopsworks, Databricks, local) you are using:


| Platform       | Spark<br/>Available | Distributed<br/>Hyperparameter<br/>Optimization | Distributed<br/>Ablation Study | Distributed<br/>Training |
|----------------|---------------------|-------------------------------------------------|--------------------------------|--------------------------|
| **Hopsworks**  | **Yes**             | Tensorflow                                      | Tensorflow                     | Tensorflow, PyTorch      |
| **Hopsworks**  | **No**              | N.A.                                            | N.A.                           | Tensorflow               |
| **Databricks** | **Yes**             | Tensorflow                                      | Tensorflow                     | Tensorflow, PyTorch      |
| **Databricks** | **No**              | N.A.                                            | N.A.                           | Tensorflow               |
| **Local**      | **Yes**             | N.A.                                            | N.A.                           | Tensorflow, Pytorch      |
| **Local**      | **No**              | N.A.                                            | N.A.                           | Tensorflow               |

Spark is available automatically when you use Hopsworks and Databricks. Locally, it is easy to 
[install and set up](https://spark.apache.org/docs/latest/api/python/getting_started/install.html).

## Preparing your project to be used with Maggy

Maggy works by defining a function that contains the training logic of your model. 

The usual workflow for a ML project will look something like this:

```py
#datasets preparation
...

#model definition
...

#model training
...

#model testing
...
```

With Maggy, the code will be like this:

```py
#datasets preparation
...

#model definition
...

def training_function(model, train_set, test_set, params):
    #model training
    ...
    
    #model testing
    ...

#configure maggy, pass training_function to maggy and run the experiment
...
```

There are three requirements for the wrapper function:

1. The function can take _model, train_set, test_set, hparams_ as arguments (they are optionals), 
plus one optional _parameter_ reporter which is needed for reporting the current metric to the experiment driver.
2. The function should return the metric that you want to optimize for. 
This should coincide with the metric being reported in the Keras callback (see next point).
3. In order to leverage on the early stopping capabilities of maggy, 
you need to make use of the maggy reporter API. By including the reporter in your training loop, 
you are telling maggy which metric to report back to the experiment driver for optimization and to check for global stopping. 
It is as easy as adding reporter.broadcast(metric=YOUR_METRIC) for example at the end of your epoch or 
batch training step and adding a reporter argument to your function signature.
If you are not writing your own training loop you can use the pre-written Keras callbacks in the maggy.callbacks module.


```py
def training_function(params):
    #datasets preparation
    ...
    
    #model definition
    ...
    
    #model training
    ...
    
    #model testing
    ...

#configure maggy, pass training_function to maggy and run the experiment
...
```

This case is the least efficient for Hyperparameter Optimization and Ablation Studies as it will process the data 
and build a model for each execution. However, it won't affect Distributed Training.


## Simple Example

[Colab Example](https://colab.research.google.com/drive/1ZoLuQbL0Il86hTPQ58ilzEfVF6MWpWRi?usp=sharing)

In this example we are using Maggy for Distributed Training of a simple CNN model for the mnist dataset.

First, we prepare the data and define the model, notice that we are not initializing it.
That's because Maggy needs the class, not an instance of it.
```py
#datasets preparation
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.reshape(x_train, (60000, 28, 28, 1))
x_test = np.reshape(x_test, (10000, 28, 28, 1))

#model definition
class SimpleCNN(tf.keras.Model):

    def __init__(self, nlayers):
        super().__init__()
        self.conv1 = keras.layers.Conv2D(28, 2, activation='relu')
        self.flatten = keras.layers.Flatten()
        self.d1 = keras.layers.Dense(32, activation='relu')
        self.d2 = keras.layers.Dense(10, activation='softmax')
        
    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = SimpleCNN #do not initialize the model, Maggy will do it for you
```

Now we wrap the trianinig logic in a function the we called _training_function_
```py
def training_function(model, train_set, test_set, hparams):
    #model training
    
    # Define training parameters
    # Hyperparamters to optimize
    nlayers = hparams['nlayers']
    learning_rate = hparams['learning_rate']
    # Fixed parameters
    num_epochs = 10
    batch_size = 256
    
    criterion = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate,momentum=0.9,decay=1e-5)
    
    model = model(nlayers = nlayers)
    
    model.compile(optimizer=optimizer, loss=criterion, metrics=["accuracy"])
    
    #model training
    model.fit(train_set[0],
              train_set[1],
              batch_size=batch_size,
              epochs=num_epochs,
              )

    # model testing    
    loss = model.evaluate(
        test_set[0],
        test_set[1],
        batch_size=32)
    
    return loss
```

Finally, we create a configuration class of the experiment we want to run (the configuration classes are 
TfDistributedConfig, TorchDistributedConfig, HyperparameterOptConfig and AblationConfig) and we launch 
the experiment using the _lagom_ function.
(Lagom is a swedish word that stands for "just the right amount").

```py
#configure maggy
config = TfDistributedConfig(name="mnist_distributed_training", 
                             model=model, 
                             train_set=(x_train, y_train), 
                             test_set=(x_test, y_test),
                             hparams=model_parameters
                            )

#run the experiment
loss = experiment.lagom(training_function, config)
```

For learning more about how to use Maggy for different use cases, you can navigate to 
[Hyperparameter Opitimization](../hpo/intro.md), [Ablation Studies](../ablation/intro.md) and 
[Training](../dist_training/intro.md).