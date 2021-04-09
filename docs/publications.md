# Publications

If you use Maggy for research, or write about Maggy please cite the following papers:

## Maggy Hyperparameter Optimization

### Maggy: Scalable Asynchronous Parallel Hyperparameter Search

#### Authors

Moritz Meister, Sina Sheikholeslami, Amir H. Payberah, Vladimir Vlassov, Jim Dowling

#### Abstract

Running extensive experiments is essential for building Machine Learning (ML) models. Such experiments usually require iterative execution of many trials with varying run times. In recent years, Apache Spark has become the de-facto standard for parallel data processing in the industry, in which iterative processes are im- plemented within the bulk-synchronous parallel (BSP) execution model. The BSP approach is also being used to parallelize ML trials in Spark. However, the BSP task synchronization barriers prevent asynchronous execution of trials, which leads to a reduced number of trials that can be run on a given computational budget. In this paper, we introduce Maggy, an open-source framework based on Spark, to execute ML trials asynchronously in parallel, with the ability to early stop poorly performing trials. In the experiments, we compare Maggy with the BSP execution of parallel trials in Spark and show that on random hyperparameter search on a con- volutional neural network for the Fashion-MNIST dataset Maggy reduces the required time to execute a fixed number of trials by 33% to 58%, without any loss in the final model accuracy.

[Download Paper](https://content.logicalclocks.com/hubfs/Maggy%20Scalable%20Asynchronous%20Parallel%20Hyperparameter%20Search.pdf)

#### Cite

```
@inproceedings{10.1145/3426745.3431338,
author = {Meister, Moritz and Sheikholeslami, Sina and Payberah, Amir H. and Vlassov, Vladimir and Dowling, Jim},
title = {Maggy: Scalable Asynchronous Parallel Hyperparameter Search},
year = {2020},
isbn = {9781450381826},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3426745.3431338},
doi = {10.1145/3426745.3431338},
booktitle = {Proceedings of the 1st Workshop on Distributed Machine Learning},
pages = {28–33},
numpages = {6},
keywords = {Scalable Hyperparameter Search, Machine Learning, Asynchronous Hyperparameter Optimization},
location = {Barcelona, Spain},
series = {DistributedML'20}
}
```

## Oblivious Training Functions

### Towards Distribution Transparency for Supervised ML With Oblivious Training Functions

#### Authors

Moritz Meister, Sina Sheikholeslami, Robin Andersson, Alexandru A. Ormenisan, Jim Dowling

#### Abstract

Building and productionizing Machine Learning (ML) models is a process of interdependent steps of iterative code updates, including exploratory model design, hyperparameter tuning, ablation experiments, and model training. Industrial-strength ML involves doing this at scale, using many compute resources, and this requires rewriting the training code to account for distribution. The result is that moving from a single host program to a cluster hinders iterative development of the software, as iterative development would require multiple versions of the software to be maintained and kept consistent. In this paper, we introduce the distribution oblivious training function as an abstraction for ML development in Python, whereby developers can reuse the same training function when running a notebook on a laptop or performing scale-out hyperparameter search and distributed training on clusters. Programs written in our framework look like industry-standard ML programs as we factor out dependencies using best-practice programming idioms (such as functions to generate models and data batches). We believe that our approach takes a step towards unifying single-host and distributed ML development.

[Download Paper](https://content.logicalclocks.com/hubfs/research/oblivious-training_mlsys20.pdf)

#### Cite

```
@inproceedings{oblivious-mlops,
author = {Meister, Moritz and Sheikholeslami, Sina and Andersson, Robin and Ormenisan, Alexandru A. and Dowling, Jim},
title = {Towards Distribution Transparency for Supervised ML With Oblivious Training Functions},
year = {2020},
booktitle = {MLSys ’20: Workshop on MLOps Systems, March 02–04},
location = {Austin, Texas, USA}
}
```
