# Quick Start

Using maggy for Distributed Training works as follows:

* Optionally, define a model generator object, similarly to what is done for Ablation Studies.
```py
class MyModel(tf.keras.Model):

    def __init__(self, ...):
        super().__init__()
        ...
        
    def call(self, ...):
        ...
    
    ...
```
* Optionally, define your train and test datasets, these will be sharded by Maggy.
```py
# Extract the data
(x_train, y_train),(x_test, y_test) = split_dataset(dataset)

# Do some preprocessing operations
...
```
* Define a training function containing the training logic.
```py
def training_function(model, train_set, test_set, hparams):
    #training and testing logic
    ...
```

* Create the configuration object and run the optimization.
```py
config = TfDistributedConfig(name="tf_test", 
                             model=model, 
                             train_set=(x_train, y_train), 
                             test_set=(x_test, y_test),
                             hparams=model_parameters),
                             ...
                             )

experiment.lagom(train_fn=training_function, config=config)
```
There are many parameters for the configuration object:
    * model: A tf.keras.Model superclass or list of them.
           Note that this has to be the class itself, not an instance.
    * train_set: The training set for the training function. If you want to load the set
            inside the training function, this can be disregarded.
    * test_set: The test set for the training function. If you want to load the set
            inside the training function, this can be disregarded.
    * process_data: The function for processing the data
    * hparams: model parameters that should be used during model initialization. Primarily
            used to give an interface for hp optimization.
    * name: Experiment name.
    * hb_interval: Heartbeat interval with which the server is polling.
    * description: A description of the experiment.