Maggy
=====

Maggy is a framework for efficient asynchronous optimization of expensive
black-box functions on top of Apache Spark. Compared to existing frameworks,
maggy is not bound to stage based optimization algorithms and therefore it is
able to make extensive use of early stopping in order to achieve efficient
resource utilization.

Right now, maggy supports asynchronous hyperparameter tuning of machine
learning and deep learning models, but other use cases include ablation studies
and asynchronous distributed training.

Moreover, it provides a developer API that allows advanced usage by
implementing custom optimization algorithms and early stopping criteria.

In order to make decisions on early stopping, the Spark executors are sending
heart beats with the current performance of the model they are training to the
maggy experiment driver which is running on the Spark driver. We call the
process of training a model with a certain hyperparameter combination a
*trial*. The experiment driver then uses all information of finished trials and
the currently running ones to check in a specified interval, which of the
trials should be stopped early.
Subsequently, the experiment driver provides a new trial to the Spark
executor.

Quick Start
-----------

To Install:

>>> pip install maggy

The programming model is that you wrap the code containing the model training
inside a wrapper function. Inside that wrapper function provide all imports and
parts that make up your experiment.

There are three requirements for this wrapper function:

1. The function should take the hyperparameters as arguments, plus one
   additional parameter reporter which is needed for reporting the current
   metric to the experiment driver.
2. The function should return the metric that you want to optimize for. This
   should coincide with the metric being reported in the Keras callback (see
   next point).
3. In order to leverage on the early stopping capabilities of maggy, you need
   to make use of the maggy reporter API. By including the reporter in your
   training loop, you are telling maggy which metric to report back to the
   experiment driver for optimization and to check for global stopping. It is
   as easy as adding reporter.broadcast(metric=YOUR_METRIC) for example at the
   end of your epoch or batch training step and adding a reporter argument to
   your function signature. If you are not writing your own training loop you
   can use the pre-written Keras callbacks in the `maggy.callbacks` module.

Sample usage:

>>> # Define Searchspace
>>> from maggy import Searchspace
>>> # The searchspace can be instantiated with parameters
>>> sp = Searchspace(kernel=('INTEGER', [2, 8]), pool=('INTEGER', [2, 8]))
>>> # Or additional parameters can be added one by one
>>> sp.add('dropout', ('DOUBLE', [0.01, 0.99]))

>>> # Define training wrapper function:
>>> def mnist(kernel, pool, dropout, reporter):
>>>     # This is your training iteration loop
>>>     for i in range(number_iterations):
>>>         ...
>>>         # add the maggy reporter to report the metric to be optimized
>>>         reporter.broadcast(metric=accuracy)
>>>         ...
>>>     # Return the same final metric
>>>     return accuracy

>>> # Launch maggy experiment
>>> from maggy import experiment
>>> result = experiment.lagom(map_fun=mnist,
>>>                            searchspace=sp,
>>>                            optimizer='randomsearch',
>>>                            direction='max',
>>>                            num_trials=15,
>>>                            name='MNIST'
>>>                           )

**lagom** is a Swedish word meaning "just the right amount". This is how maggy
uses your resources.

MNIST Example
-------------

For a full MNIST example with random search using Keras,
see the Jupyter Notebook in the `examples` folder.

Documentation
-------------

API documentation is available `here <https://maggy.readthedocs.io/en/latest/>`_.
