# API

The `GenericMethod` defines the interface for common quantification and unfolding algorithms. Most importantly, this interface consists of their `fit` and `predict` methods.

Instances of popular quantification and unfolding algorithms are created through more specified constructors, which are detailed below.

```{eval-rst}
.. autoclass:: qunfold.GenericMethod
   :members:
```

## Classify and count

```{eval-rst}
.. autofunction:: qunfold.ACC

.. autofunction:: qunfold.PACC
```
