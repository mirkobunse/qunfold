# API

The `AbstractMethod` defines the interface of all quantification and unfolding algorithms. Most importantly, this interface consists of their `fit` and `predict` methods.

```{eval-rst}
.. autoclass:: qunfold.AbstractMethod
   :members:
```

Instances of many [](#popular-algorithms) for quantification and unfolding are created through the corresponding constructors. However, you can also define your own quantification method as a `LinearMethod` that combines an arbitrary choice of [](#losses), [](#regularizers) and [](#data-representations).

```{eval-rst}
.. autoclass:: qunfold.LinearMethod
   :members:
```


## Popular algorithms

We categorize existing, well-known quantification and unfolding algorithms into [](#classify-and-count) methods, [](#distribution-matching) methods, and [](#unfolding) methods. Each of these methods consists of a fixed combination of [](#losses), [](#regularizers), and [](#data-representations).


### Classify and count

```{eval-rst}
.. autoclass:: qunfold.ACC

.. autoclass:: qunfold.PACC
```


### Distribution matching

```{eval-rst}
.. autoclass:: qunfold.EDx

.. autoclass:: qunfold.EDy

.. autoclass:: qunfold.HDx

.. autoclass:: qunfold.HDy

.. autoclass:: qunfold.KMM
```


### Unfolding

```{eval-rst}
.. autoclass:: qunfold.RUN
```


### Methods beyond systems of linear equations

Not all quantification algorithms make predictions by solving systems of linear equations. Instead, the following methods maximize the likelihood of the prediction directly.

```{eval-rst}
.. autoclass:: qunfold.LikelihoodMaximizer

.. autoclass:: qunfold.ExpectationMaximizer
```


## Losses

```{eval-rst}
.. autoclass:: qunfold.LeastSquaresLoss

.. autoclass:: qunfold.EnergyLoss

.. autoclass:: qunfold.HellingerSurrogateLoss

.. autoclass:: qunfold.BlobelLoss

.. autoclass:: qunfold.CombinedLoss
```

```{hint}
You can use the `CombinedLoss` to create arbitrary, weighted sums of losses and regularizers.
```


## Regularizers

```{eval-rst}
.. autofunction:: qunfold.TikhonovRegularized

.. autoclass:: qunfold.TikhonovRegularization
```


## Data representations

```{eval-rst}
.. autoclass:: qunfold.ClassRepresentation

.. autoclass:: qunfold.DistanceRepresentation

.. autoclass:: qunfold.HistogramRepresentation

.. autoclass:: qunfold.EnergyKernelRepresentation

.. autoclass:: qunfold.GaussianKernelRepresentation

.. autoclass:: qunfold.LaplacianKernelRepresentation

.. autoclass:: qunfold.GaussianRFFKernelRepresentation

.. autoclass:: qunfold.OriginalRepresentation
```


## Utilities

The following classes provide functionalities that go beyond the composition of quantification methods.

### QuaPy

The `qunfold.quapy` module allows you to wrap any quantification method for being used in [QuaPy](https://github.com/HLT-ISTI/QuaPy).

```{eval-rst}
.. autoclass:: qunfold.quapy.QuaPyWrapper
```

### Cross-validated training

The `qunfold.sklearn` module allows you to train classification-based quantification methods through cross-validation. Importing this module requires [scikit-learn](https://scikit-learn.org/stable/) to be installed.

```{eval-rst}
.. autoclass:: qunfold.sklearn.CVClassifier
```

```{hint}
If you use a bagging classifier (like random forests) with `oob_score=True`, you do not need to use cross-validation. Instead, the quantification method is then trained on the out-of-bag predictions of the bagging classifier.
```
