# Quick Start

Ablation studies have become best practice in Machine Learning research as they provide insights into the relative 
contribution of the different architectural and regularization components to the performance of models. 
An ablation study consists of several trials, where one trial could be, 
e.g., removing the last convolutional layer of a CNN model, retraining the model, 
and observing the resulting performance. 
However, as machine learning architectures become ever deeper and data sizes keep growing, 
there is an explosion in the number of different architecture combinations that need to be evaluated to understand 
their relative performance. 

Our framework provides a declarative way to define ablation experiments for deep learning model architectures 
and training datasets, in a way that eliminates the need for maintaining redundant copies of code for an ablation study.
Furthermore, our framework enables parallel execution of ablation trials without requiring the developers to 
modify their code, which leads to shorter study times and better resource utilization. 

## Simple Example
