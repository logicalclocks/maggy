Maggy
=====

|Downloads| |PypiStatus| |PythonVersions| |Docs| |CodeStyle|

Maggy is a framework for efficient asynchronous optimization of expensive
black-box functions on top of Apache Spark. Compared to existing frameworks,
maggy is not bound to stage based optimization algorithms and therefore it is
able to make extensive use of early stopping in order to achieve efficient
resource utilization.

For a video describing Maggy, see `this talk at the Spark/AI Summit <https://www.youtube.com/watch?v=0Hd1iYEL03w>`_.

Right now, maggy supports asynchronous hyperparameter tuning of machine
learning and deep learning models, and ablation studies on neural network
layers as well as input features.

Moreover, it provides a developer API that allows advanced usage by
implementing custom optimization algorithms and early stopping criteria.

To accomodate asynchronous algorithms, support for communication between the
Driver and Executors via RPCs through Maggy was added. The Optimizer that guides
hyperparameter search is located on the Driver and it assigns trials to
Executors. Executors periodically send back to the Driver the current
performance of their trial, and the Optimizer can decide to early-stop any
ongoing trial and send the Executor a new trial instead.

Quick Start
-----------

To Install:

>>> pip install maggy

The programming model consists of wrapping the code containing the model training
inside a function. Inside that wrapper function provide all imports and
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
>>> result = experiment.lagom(train_fn=mnist,
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

Read our `blog post <https://www.logicalclocks.com/blog/scaling-machine-learning-and-deep-learning-with-pyspark-on-hopsworks>`_ for more details.

API documentation is available `here <https://maggy.readthedocs.io/en/latest/>`_.

.. |Downloads| image:: https://pepy.tech/badge/maggy/month
   :target: https://pepy.tech/project/maggy
.. |PypiStatus| image:: https://img.shields.io/pypi/v/maggy?color=blue
    :target: https://pypi.org/project/maggy
.. |PythonVersions| image:: https://img.shields.io/pypi/pyversions/maggy.svg
    :target: https://pypi.org/project/maggy
.. |Docs| image:: https://img.shields.io/readthedocs/maggy
    :target: https://maggy.readthedocs.io/en/latest/
.. |CodeStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
