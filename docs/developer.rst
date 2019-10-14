Maggy Developer API
===================

As a developer you have the possibility to implement your custom optimizers
or ablators. For that you can implement an abstract method, which you can then
pass as an argument when launching the experiment. For examples, please look at
existing optimizers and ablators.

maggy.optimizer module
-----------------------

.. autoclass:: maggy.optimizer.AbstractOptimizer
    :members:

maggy.ablation.ablator module
-----------------------------

.. autoclass:: maggy.ablation.ablator.abstractablator.AbstractAblator
    :members: