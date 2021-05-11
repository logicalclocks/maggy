<p align="center">
  <a href="https://github.com/logicalclocks/maggy">
    <img src="https://raw.githubusercontent.com/moritzmeister/maggy/mkdocs/docs/assets/images/maggy.png" width="320" alt="Maggy">
  </a>
</p>

<p align="center">
  <a href="https://community.hopsworks.ai"><img
    src="https://img.shields.io/discourse/users?label=Hopsworks%20Community&server=https%3A%2F%2Fcommunity.hopsworks.ai"
    alt="Hopsworks Community"
  /></a>
    <a href="https://maggy.ai"><img
    src="https://img.shields.io/badge/docs-MAGGY-orange"
    alt="Maggy Documentation"
  /></a>
  <a href="https://pypi.org/project/maggy/"><img
    src="https://img.shields.io/pypi/v/maggy?color=blue"
    alt="PyPiStatus"
  /></a>
  <a href="https://pepy.tech/project/maggy/month"><img
    src="https://pepy.tech/badge/maggy/month"
    alt="Downloads"
  /></a>
  <a href="https://github.com/psf/black"><img
    src="https://img.shields.io/badge/code%20style-black-000000.svg"
    alt="CodeStyle"
  /></a>
  <a><img
    src="https://img.shields.io/pypi/l/maggy?color=green"
    alt="License"
  /></a>
</p>

Maggy is a framework for **distribution transparent** machine learning experiments on [Apache Spark](https://spark.apache.org/).
In this post, we introduce a new unified framework for writing core ML training logic as **oblivious training functions**.
Maggy enables you to reuse the same training code whether training small models on your laptop or reusing the same code to scale out hyperparameter tuning or distributed deep learning on a cluster.
Maggy enables the replacement of the current waterfall development process for distributed ML applications, where code is rewritten at every stage to account for the different distribution context.

<p align="center">
  <figure>
    <a href="https://github.com/logicalclocks/maggy">
      <img src="https://raw.githubusercontent.com/moritzmeister/maggy/mkdocs/docs/assets/images/firstgraph.png" alt="Maggy">
    </a>
    <figcaption>Maggy uses the same distribution transparent training function in all steps of the machine learning development process.</figcaption>
  </figure>
</p>

## Quick Start

Maggy uses PySpark as an engine to distribute the training processes. To get started, install Maggy in the Python environment used by your Spark Cluster, or install Maggy in your local Python environment with the `'spark'` extra, to run on Spark in local mode:

```python
pip install maggy
```

The programming model consists of wrapping the code containing the model training
inside a function. Inside that wrapper function provide all imports and
parts that make up your experiment.

Single run experiment:

```python
def train_fn():
    # This is your training iteration loop
    for i in range(number_iterations):
        ...
        # add the maggy reporter to report the metric to be optimized
        reporter.broadcast(metric=accuracy)
         ...
    # Return metric to be optimized or any metric to be logged
    return accuracy

from maggy import experiment
result = experiment.lagom(train_fn=train_fn, name='MNIST')
```

**lagom** is a Swedish word meaning "just the right amount". This is how MAggy
uses your resources.


## Documentation

Full documentation is available at [maggy.ai](https://maggy.ai/)

## Contributing

There are various ways to contribute, and any contribution is welcome, please follow the
CONTRIBUTING guide to get started.

## Issues

Issues can be reported on the official [GitHub repo](https://github.com/logicalclocks/maggy/issues) of Maggy.

## Citation

Please see our publications on [maggy.ai](https://maggy.ai/publications) to find out how to cite our work.
