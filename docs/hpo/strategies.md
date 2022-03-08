# Quick Start

Using maggy for Hyperparameter Optimization (HPO) works as follows:

* Define a training function containing the training logic.
```py
def training_function(hparams):
    #training and testing logic
    ...
```

* Define a search space, containing the HPs we want to optimize, their type and range.
```py
#define the hyperparemeters to optimize, together with their possible values
sp = Searchspace(kernel=('INTEGER', [2, 8]), pool=('INTEGER', [2, 8]), dropout=('DOUBLE', [0.01, 0.99]))
```

* Create the configuration object and run the optimization.
```py
config = OptimizationConfig(num_trials=4, 
                            optimizer="randomsearch", 
                            searchspace=sp, 
                            direction="max", 
                            es_interval=1, 
                            es_min=5, 
                            name="hp_tuning_test")

experiment.lagom(train_fn=training_function, config=config)
```
There are many parameters for the configuration object:
    * num_trials: Controls how many seperate runs are conducted during the hp search.
    * optimizer: Optimizer type for searching the hp searchspace.
    * searchspace: A Searchspace object configuring the names, types and ranges of hps.
    * optimization_key: Name of the metric to use for hp search evaluation.
    * direction: Direction of optimization.
    * es_interval: Early stopping polling frequency during an experiment run.
    * es_min: Minimum number of experiments to conduct before starting the early stopping
    mechanism. Useful to establish a baseline for performance estimates.
    * es_policy: Early stopping policy which formulates a rule for triggering aborts.
    * name: Experiment name.
    * description: A description of the experiment.
    * hb_interval: Heartbeat interval with which the server is polling.
    * model: The class of the model to be used in the training function.
    * train_set: The train_set to be used in the training function.
    * test_set: The test_set to be used in the training function.

# Strategies

### Random Search

With Random Search, the HPs are selected randomly within the search space defined. The search space is defined 
depending on how many trials (_num_trials_) you choose. 

In the following example, _num_trials_ is set to 4, therefore, Maggy will choose randomly 4 combinations of kernel, 
pool and dropout values.
```py
def training_function(hparams):
    #training and testing logic
    ...
#define the hyperparemeters to optimize, together with their possible values
sp = Searchspace(kernel=('INTEGER', [2, 8]), pool=('INTEGER', [2, 8]), dropout=('DOUBLE', [0.01, 0.99]))

config = OptimizationConfig(num_trials=4, 
                            optimizer="randomsearch", 
                            searchspace=sp, 
                            direction="max", 
                            es_interval=1, 
                            es_min=5, 
                            name="hp_tuning_test")

#run optimization
result = experiment.lagom(train_fn=training_function, config=config)
```
### Grid Search

```py
def training_function():
    #training and testing logic
    ...
#define the hyperparemeters to optimize, together with their possible values
sp = Searchspace(kernel=('INTEGER', [2, 8]), pool=('INTEGER', [2, 8]), dropout=('DOUBLE', [0.01, 0.99]))

config = OptimizationConfig(num_trials=4, 
                            optimizer="gridsearch", 
                            searchspace=sp, 
                            direction="max", 
                            es_interval=1, 
                            es_min=5, 
                            name="hp_tuning_test")

#run optimization
result = experiment.lagom(train_fn=training_function, config=config)
```

### Asynchronous Successive Halving Algorithm (ASHA)

This strategy is a combination of random search and early stopping. 
ASHA tackles large-scale hyperparameter optimization problems, and it is especially useful for challenges that need a
high number of parallelism (i.e. there are a lot of HPs and a lot of workers are available).

```py
def training_function():
    #training and testing logic
    ...
#define the hyperparemeters to optimize, together with their possible values
sp = Searchspace(kernel=('INTEGER', [2, 8]), pool=('INTEGER', [2, 8]), dropout=('DOUBLE', [0.01, 0.99]))

config = OptimizationConfig(num_trials=4, 
                            optimizer='asha', 
                            searchspace=sp, 
                            direction="max", 
                            es_interval=1, 
                            es_min=5, 
                            name="hp_tuning_test")

experiment.lagom(train_fn=training_function, config=config)
```

you can define custom ASHA optimizers by setting 3 parameters: _reduction_factor, resource_min_ and _resource_max_.
The standard values are 2, 1, and 4, respectively.
To use custom values, import the class _Asha_ from _maggy.optimizer_ and create the object with custom 
parameters.

```py
from maggy.optimizer import Asha

asha = Asha(3,1,10)
config = OptimizationConfig(..., 
                            optimizer=asha, 
                            ...)
```

### Bayesian Optimization

WIth Bayesian Optimization (BO), the HPs are chosen based on the space of the HPs. 
In order to do that, the algorithm infer a function of the HPs in order to optimize the cost function of a given model.

There are two different BO methods available in Maggy, namely Gaussian Process (GP) and Tree Parzen Estimators (TPE).
The GP is a tool used to infer the value of a function in which predictions follow a normal distribution. 
We use that set of predictions and pick new points where we should evaluate next. From that new point, we add it to 
the samples and re-build the Gaussian Process with that new information… 
We keep doing this until we reach the maximal number of iterations or the limit time for example.
TPE is an iterative process that uses history of evaluated hyperparameters to create probabilistic model, 
which is used to suggest next set of hyperparameters to evaluate.


```py
def training_function():
    #training and testing logic
    ...
#define the hyperparemeters to optimize, together with their possible values
sp = Searchspace(kernel=('INTEGER', [2, 8]), pool=('INTEGER', [2, 8]), dropout=('DOUBLE', [0.01, 0.99]))

config = OptimizationConfig(num_trials=4, 
                            optimizer='gp', #or 'tpe' 
                            searchspace=sp, 
                            direction="max", 
                            es_interval=1, 
                            es_min=5, 
                            name="hp_tuning_test")

experiment.lagom(train_fn=training_function, config=config)
```