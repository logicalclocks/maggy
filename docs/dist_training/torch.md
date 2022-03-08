# Quick Start

Maggy enables you to train with Microsoft’s DeepSpeed ZeRO optimizer. Since DeepSpeed does not follow the common 
PyTorch programming model, Maggy is unable to provide full distribution transparency to the user. 
This means that if you want to use DeepSpeed for your training, you will have to make small changes 
to your code. In this notebook, we will show you what exactly you have to change in order to make 
DeepSpeed run with Maggy.

* First off, we have to define our model as we did for TensorFlow and Ablation studies.
```py
class MyModel(torch.nn.Module):
    
    def __init__(self, ...):
        super().__init__(...)
        ...
        
    def forward(self, ...):
       ...
```

* There are a few minor changes that have to be done in order to train with DeepSpeed: - There is no need for an 
optimizer anymore. You can configure your optimizer later in the DeepSpeed config. - DeepSpeed’s ZeRO requires you to 
use FP16 training. Therefore, convert your data to half precision! - The backward call is not executed on the loss, 
but on the model (```model.backward(loss)``` instead of ```loss.backward()```). - 
The step call is not executed on the optimizer, 
but also on the model (```model.step()``` instead of ```optimizer.step()```). - 
As we have no optimizer anymore, there is also 
no need to call ```optimizer.zero_grad()```. 
You do not have to worry about the implementation of these calls, 
Maggy configures your model at runtime to act as a DeepSpeed engine.
```py
def train_fn(...):
    ...
```

* In order to use DeepSpeed’s ZeRO, the deepspeed backend has to be chosen. This 
backend also requires its own config. You can read a full specification of the possible settings 
[here](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training).
```py
ds_config = {"train_micro_batch_size_per_gpu": 1,
 "gradient_accumulation_steps": 1,
 "optimizer": {"type": "Adam", "params": {"lr": 0.1}},
 "fp16": {"enabled": True},
 "zero_optimization": {"stage": 2},
}

config = TorchDistributedConfig(module=MyModel, backend="deepspeed", deepspeed_config=ds_config, ...)
```

* Start the training with ```lagom()```
```py
result = experiment.lagom(train_fn, config)
```