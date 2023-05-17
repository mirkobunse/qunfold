# API

The `GenericMethod` defines the interface for many common quantification and unfolding algorithms. Most importantly, this interface consists of their `fit` and `predict` methods.

Instances of [](#popular-algorithms) for quantification and unfolding are created through specialized constructors. However, you can also define your own quantification algorithm as a `GenericMethod` that combines an arbitrary choice of [](#losses), [](#regularizers) and [](#feature-transformations).

```{eval-rst}
.. autoclass:: qunfold.GenericMethod
   :members:
```


## Popular algorithms

We categorize existing, well-known quantification and unfolding algorithms into [](#classify-and-count) methods, [](#distribution-matching) methods, and [](#unfolding) methods. Each of these methods consists of a fixed combination of [](#losses), [](#regularizers), and [](#feature-transformations).


### Classify and count

```{eval-rst}
.. autoclass:: qunfold.ACC

.. autoclass:: qunfold.PACC
```


### Distribution matching

```{eval-rst}
.. autoclass:: qunfold.HDx

.. autoclass:: qunfold.HDy
```


### Unfolding

```{eval-rst}
.. autoclass:: qunfold.RUN
```


## Losses

```{eval-rst}
.. autoclass:: qunfold.LeastSquaresLoss

.. autoclass:: qunfold.BlobelLoss

.. autoclass:: qunfold.HellingerLoss

.. autoclass:: qunfold.CombinedLoss
```

```{hint}
You can use the `CombinedLoss` to create arbitrary, weighted sums of losses and regularizers.
```


## Regularizers

```{eval-rst}
.. autoclass:: qunfold.TikhonovRegularization

.. autoclass:: qunfold.TikhonovRegularized
```


## Feature transformations

```{eval-rst}
.. autoclass:: qunfold.ClassTransformer

.. autoclass:: qunfold.HistogramTransformer
```
